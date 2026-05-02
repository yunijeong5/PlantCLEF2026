"""
Unified PlantCLEF 2026 inference pipeline.

Combines:
  • Paper 1 pre-processing: JPEG tile compression + multi-scale tiling
  • Paper 2 post-processing: Bayesian prior + geographical species filter

Every stage is optional and controlled by CLI flags, enabling ablation studies
without re-running expensive feature extraction (all intermediate results are
cached to disk).

Example invocations
-------------------
# Full pipeline (paper1 + paper2):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_full.csv

# Multi-scale tiling only (no prior, no geo filter):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_tiling_only.csv \\
    --no-bayesian-prior --no-geo-filter

# Paper2 baseline (4×4 tiling + prior + geo filter):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_paper2.csv \\
    --scales 4 --aggregation mean

# Scale-1 sanity check (no tiling, no extras):
python -m pipeline.run_pipeline \\
    --images-dir data/test/images \\
    --output output/submission_baseline.csv \\
    --scales 1 --no-jpeg --no-bayesian-prior --no-geo-filter
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .aggregation import aggregate
from .config import PipelineConfig
from .features import get_all_logits, load_cached_logits_scale1
from .geo_filter import apply_geo_filter, load_or_build_geo_mask
from .model import load_model
from .prior import (
    apply_prior,
    compute_prior_from_logits,
    load_prior_from_file,
    save_priors,
)
from .submission import load_class_names, write_submission


# ── helpers ───────────────────────────────────────────────────────────────────

def _image_paths(images_dir: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in sorted(Path(images_dir).iterdir()) if p.suffix in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir!r}")
    return paths


def _ensure_scale1_in_scales(scales: List[int]) -> List[int]:
    """Add scale=1 if needed for prior computation; return augmented list."""
    if 1 not in scales:
        return scales + [1]
    return scales


# ── pipeline ──────────────────────────────────────────────────────────────────

def run(cfg: PipelineConfig) -> Dict[str, List[int]]:
    t0 = time.time()
    img_paths = _image_paths(cfg.images_dir)
    stems = [p.stem for p in img_paths]
    print(f"Found {len(img_paths)} test images.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
    class_ids = load_class_names(cfg.class_mapping_file)  # index → species_id

    # ── Decide which scales to run ────────────────────────────────────────────
    # If Bayesian prior is requested and no pre-computed file is given, we need
    # scale-1 logits to compute the prior.  Add scale=1 if absent.
    run_scales = list(cfg.scales)
    extra_scale1 = False
    if cfg.use_bayesian_prior and cfg.prior_data_path is None and 1 not in run_scales:
        run_scales = _ensure_scale1_in_scales(run_scales)
        extra_scale1 = True   # we added it only for prior computation

    # ── Geo mask ──────────────────────────────────────────────────────────────
    geo_mask: Optional[np.ndarray] = None
    if cfg.use_geo_filter:
        geo_mask = load_or_build_geo_mask(
            cfg.training_metadata_csv,
            cfg.num_classes,
            cfg.cache_dir,
            cfg.geo_ref_lat,
            cfg.geo_ref_lon,
            cfg.geo_country_boxes,
        )

    # ── Feature extraction (all images, all scales) ───────────────────────────
    jpeg_desc = (f"JPEG {cfg.jpeg_subsampling} q{cfg.jpeg_quality}"
                 if cfg.use_jpeg_compression else "no JPEG compression")
    print(f"Extracting features for scales {run_scales} ({jpeg_desc})...")

    # all_logits[i] = list of arrays [S², C] for image i (one per scale)
    all_logits_per_image: List[List[np.ndarray]] = []
    for idx, (img_path, stem) in enumerate(zip(img_paths, stems)):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(img_paths)}]  {stem}  ({elapsed:.0f}s elapsed)")

        img = Image.open(img_path).convert("RGB")
        scale_logits = get_all_logits(
            img, stem, run_scales, model, cfg.cache_dir,
            cfg.tile_size, cfg.batch_size,
            cfg.use_jpeg_compression, cfg.jpeg_quality, cfg.jpeg_subsampling,
        )
        all_logits_per_image.append(scale_logits)

    # ── Bayesian prior ────────────────────────────────────────────────────────
    priors = None
    if cfg.use_bayesian_prior:
        if cfg.prior_data_path:
            print(f"Loading pre-computed priors from {cfg.prior_data_path}")
            priors = load_prior_from_file(cfg.prior_data_path)
        else:
            # Compute from scale-1 logits (index in run_scales)
            s1_idx = run_scales.index(1)
            scale1_logits = np.concatenate(
                [img_logits[s1_idx] for img_logits in all_logits_per_image], axis=0
            )  # [N, C]
            print("Computing Bayesian priors from scale-1 predictions...")
            priors = compute_prior_from_logits(stems, scale1_logits)
            save_priors(priors, cfg.cache_dir)

    # ── Aggregate + post-process ───────────────────────────────────────────────
    print("Aggregating tile predictions...")
    results: Dict[str, List[int]] = {}

    for stem, img_scale_logits in zip(stems, all_logits_per_image):
        # Select only the user-requested scales (drop extra scale-1 if added)
        if extra_scale1:
            selected = [
                logits for s, logits in zip(run_scales, img_scale_logits) if s != 1
            ]
        else:
            selected = img_scale_logits

        # Concatenate all tiles across scales: [total_tiles, C]
        tile_logits = np.concatenate(selected, axis=0)

        # Aggregate across tiles → [C]
        image_probs = aggregate(tile_logits, cfg.aggregation, cfg.topk_mean_k, cfg.vote_k)

        # Apply Bayesian prior (after aggregation — equivalent to before, more efficient)
        if priors is not None:
            image_probs = apply_prior(image_probs, stem, priors)

        # Geographical filter
        if geo_mask is not None:
            image_probs = apply_geo_filter(image_probs, geo_mask)

        # Top-K with minimum score threshold
        order = np.argsort(image_probs)[::-1]
        top_ids = [
            class_ids[i]
            for i in order
            if image_probs[i] >= cfg.min_score
        ][: cfg.top_k]
        results[stem] = top_ids

    total = time.time() - t0
    print(f"Done. {len(results)} images processed in {total:.1f}s "
          f"({total / len(results):.2f}s/image).")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> PipelineConfig:
    defaults = PipelineConfig()

    p = argparse.ArgumentParser(
        description="PlantCLEF 2026 unified inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--images-dir",  default=defaults.images_dir)
    p.add_argument("--output",      default=str(Path(defaults.output_dir) / "submission.csv"),
                   help="Path of the output submission CSV.")
    p.add_argument("--cache-dir",   default=defaults.cache_dir,
                   help="Directory for cached features and logits.")

    # Model
    p.add_argument("--model-path",         default=defaults.model_path)
    p.add_argument("--class-mapping-file", default=defaults.class_mapping_file)
    p.add_argument("--batch-size",         type=int, default=defaults.batch_size)

    # Pre-processing (paper 1)
    p.add_argument("--jpeg",    dest="use_jpeg_compression", action="store_true",  default=True)
    p.add_argument("--no-jpeg", dest="use_jpeg_compression", action="store_false")
    p.add_argument("--jpeg-quality", type=int, default=defaults.jpeg_quality)
    p.add_argument(
        "--jpeg-subsampling",
        default=defaults.jpeg_subsampling,
        choices=["4:4:4", "4:2:2", "4:2:0", "4:1:1"],
        help=(
            "Chroma subsampling for the JPEG round-trip. "
            "Paper 1 best: '4:2:2' at quality 85, or '4:1:1' at quality 94 "
            "(the latter requires:  pip install PyTurboJPEG)."
        ),
    )
    p.add_argument(
        "--scales", type=int, nargs="+", default=defaults.scales,
        help="Tiling scales, e.g. --scales 6 5 4 3 2 1  or  --scales 4  or  --scales 1",
    )

    # Aggregation
    p.add_argument(
        "--aggregation", choices=["max", "mean", "topk_mean", "vote"], default=defaults.aggregation,
        help="Tile aggregation method.",
    )
    p.add_argument("--topk-mean-k", type=int, default=defaults.topk_mean_k,
                   help="k for topk_mean aggregation.")
    p.add_argument("--vote-k", type=int, default=defaults.vote_k,
                   help="Top-k species each tile votes for (vote aggregation).")

    # Post-processing (paper 2)
    p.add_argument("--bayesian-prior",    dest="use_bayesian_prior", action="store_true",  default=True)
    p.add_argument("--no-bayesian-prior", dest="use_bayesian_prior", action="store_false")
    p.add_argument("--prior-data-path",   default=defaults.prior_data_path,
                   help="Pre-computed cluster priors .npy file, shape [NUM_CLUSTERS, num_classes]. "
                        "If omitted, priors are computed from scale-1 predictions.")

    p.add_argument("--geo-filter",    dest="use_geo_filter", action="store_true",  default=True)
    p.add_argument("--no-geo-filter", dest="use_geo_filter", action="store_false")
    p.add_argument("--training-metadata-csv", default=defaults.training_metadata_csv)

    # Submission
    p.add_argument(
        "--top-k", type=int, nargs="+", default=[defaults.top_k],
        help="One or more K values. When multiple are given, writes one CSV per K "
             "(use {k} in --output as a template, e.g. output/sub_{k}.csv, or names "
             "are auto-generated as <stem>_topk<K>.csv). The pipeline runs once.",
    )
    p.add_argument("--min-score", type=float, default=defaults.min_score)

    a = p.parse_args()
    top_k_list = sorted(set(a.top_k))  # deduplicate, ascending

    cfg = PipelineConfig(
        images_dir              = a.images_dir,
        output_dir              = str(Path(a.output).parent),
        cache_dir               = a.cache_dir,
        model_path              = a.model_path,
        class_mapping_file      = a.class_mapping_file,
        batch_size              = a.batch_size,
        use_jpeg_compression    = a.use_jpeg_compression,
        jpeg_quality            = a.jpeg_quality,
        jpeg_subsampling        = a.jpeg_subsampling,
        scales                  = a.scales,
        aggregation             = a.aggregation,
        topk_mean_k             = a.topk_mean_k,
        vote_k                  = a.vote_k,
        use_bayesian_prior      = a.use_bayesian_prior,
        prior_data_path         = a.prior_data_path,
        use_geo_filter          = a.use_geo_filter,
        training_metadata_csv   = a.training_metadata_csv,
        top_k                   = max(top_k_list),  # run with largest K; main() slices
        min_score               = a.min_score,
    )
    # Stored for main() — not part of PipelineConfig dataclass
    cfg._output_csv = a.output
    cfg._top_k_list = top_k_list
    return cfg


def _format_output_path(template: str, k: int, multi: bool) -> str:
    """Return output path for a given K value.

    - If template contains '{k}', substitute it.
    - If multi=True and no '{k}', insert '_topk{k}' before the extension.
    - Otherwise return template unchanged (single-K, backward-compatible).
    """
    if "{k}" in template:
        return template.format(k=k)
    if multi:
        p = Path(template)
        return str(p.parent / f"{p.stem}_topk{k}{p.suffix}")
    return template


def main() -> None:
    cfg = _parse_args()
    top_k_list: List[int] = cfg._top_k_list  # set by _parse_args
    results = run(cfg)  # runs with max(top_k_list); results already sliced to that max

    multi = len(top_k_list) > 1
    for k in top_k_list:
        sliced = {stem: species[:k] for stem, species in results.items()}
        out_path = _format_output_path(cfg._output_csv, k, multi)
        write_submission(sliced, out_path)
        if multi:
            print(f"Written top-{k}: {out_path}")


if __name__ == "__main__":
    main()
