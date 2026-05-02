"""
HPC smoke-test for the PlantCLEF 2026 pipeline.

Run this in an interactive salloc session before submitting sbatch:

    python test_pipeline_hpc.py           # all tests
    python test_pipeline_hpc.py --fast    # skip full multi-scale (scales=[4] only)

Each test prints PASS / FAIL / SKIP with timing.
A summary at the end shows peak GPU VRAM and overall status.

Exit code: 0 if all non-skipped tests pass, 1 otherwise.
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent


# ── helpers ───────────────────────────────────────────────────────────────────

PASS  = "\033[32mPASS\033[0m"
FAIL  = "\033[31mFAIL\033[0m"
SKIP  = "\033[33mSKIP\033[0m"
_results: list[tuple[str, str, float]] = []   # (name, status, seconds)

def _run(name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), print result, record outcome."""
    t0 = time.time()
    try:
        msg = fn(*args, **kwargs)
        elapsed = time.time() - t0
        status = PASS
        label = PASS
    except _SkipTest as e:
        elapsed = time.time() - t0
        msg = str(e)
        status = label = SKIP
    except Exception as e:
        elapsed = time.time() - t0
        msg = str(e)
        status = FAIL
        label = FAIL
    _results.append((name, label, elapsed))
    detail = f"  {msg}" if msg else ""
    print(f"[{status}] {name} ({elapsed:.1f}s){detail}")
    return label


class _SkipTest(Exception):
    pass


# ── test functions ─────────────────────────────────────────────────────────────

def test_imports():
    import numpy, torch, PIL, scipy, pandas, timm  # noqa: F401
    import torchvision  # noqa: F401
    from pipeline import (  # noqa: F401
        aggregation, compression, config, features,
        geo_filter, model, prior, submission, tiling,
    )
    import torch
    return (f"torch {torch.__version__}, "
            f"timm {__import__('timm').__version__}, "
            f"PIL {PIL.__version__}")


def test_gpu():
    import torch
    if not torch.cuda.is_available():
        raise _SkipTest("No CUDA device — running on CPU")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    total_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
    return f"{n}× {names[0]}, {total_mb} MB total VRAM"


def test_tiling():
    from PIL import Image
    from pipeline.tiling import extract_tiles

    img = Image.open(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg").convert("RGB")
    expected = {1: 1, 2: 4, 4: 16, 6: 36}
    for scale, n_tiles in expected.items():
        tiles = extract_tiles(img, scale, tile_size=518)
        assert len(tiles) == n_tiles, f"scale={scale}: got {len(tiles)} tiles, want {n_tiles}"
        for t in tiles:
            assert t.size == (518, 518), f"scale={scale}: tile size {t.size}"
    return "scales [1,2,4,6] → [1,4,16,36] tiles of 518×518"


def test_jpeg_compression():
    from PIL import Image
    from pipeline.compression import jpeg_compress

    img = Image.open(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg").convert("RGB")
    for mode in ["4:4:4", "4:2:2", "4:2:0"]:
        out = jpeg_compress(img, quality=85, subsampling=mode)
        assert out.size == img.size and out.mode == "RGB", f"{mode} failed"
    return "4:4:4 / 4:2:2 / 4:2:0 OK"


def test_jpeg_411():
    try:
        from turbojpeg import TurboJPEG  # noqa: F401
    except ImportError:
        raise _SkipTest("PyTurboJPEG not installed — 4:1:1 unavailable")
    from PIL import Image
    from pipeline.compression import jpeg_compress
    img = Image.open(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg").convert("RGB")
    out = jpeg_compress(img, quality=94, subsampling="4:1:1")
    assert out.size == img.size and out.mode == "RGB"
    return "4:1:1 via TurboJPEG OK"


def test_model_load():
    from pipeline.model import load_model
    from pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.model_path}")

    import torch
    t0 = time.time()
    model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
    load_s = time.time() - t0

    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Check head output dimension
    import numpy as np
    dummy = np.zeros((1, 768), dtype=np.float32)
    logits = __import__("pipeline.model", fromlist=["features_to_logits_batched"]).features_to_logits_batched(model, dummy)
    assert logits.shape == (1, cfg.num_classes), f"head shape mismatch: {logits.shape}"

    return f"{n_params:.1f}M params on {device}, load={load_s:.1f}s"


def test_inference_scale1(tmp_dir: str):
    """One image, scale=1 (1 tile). Measures GPU VRAM peak."""
    import torch
    from PIL import Image
    from pipeline.config import PipelineConfig
    from pipeline.model import load_model
    from pipeline.features import get_all_logits

    cfg = PipelineConfig()
    if not Path(cfg.model_path).exists():
        raise _SkipTest("Checkpoint not found — skipping inference test")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
    img = Image.open(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg").convert("RGB")

    t0 = time.time()
    logits_list = get_all_logits(
        img, "sample", [1], model, tmp_dir,
        tile_size=cfg.tile_size, batch_size=cfg.batch_size,
        use_jpeg_compression=True, jpeg_quality=85, jpeg_subsampling="4:2:2",
    )
    elapsed = time.time() - t0

    assert logits_list[0].shape == (1, cfg.num_classes), f"bad shape: {logits_list[0].shape}"

    vram_msg = ""
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() // 1024**2
        vram_msg = f", peak VRAM={peak_mb} MB"

    return f"logits shape={logits_list[0].shape}, {elapsed:.1f}s{vram_msg}"


def test_inference_scale4(tmp_dir: str):
    """One image, scale=4 (16 tiles). Representative of multi-scale cost."""
    import torch
    from PIL import Image
    from pipeline.config import PipelineConfig
    from pipeline.model import load_model
    from pipeline.features import get_all_logits

    cfg = PipelineConfig()
    if not Path(cfg.model_path).exists():
        raise _SkipTest("Checkpoint not found — skipping inference test")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
    img = Image.open(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg").convert("RGB")

    t0 = time.time()
    logits_list = get_all_logits(
        img, "sample", [4], model, tmp_dir,
        tile_size=cfg.tile_size, batch_size=cfg.batch_size,
        use_jpeg_compression=True, jpeg_quality=85, jpeg_subsampling="4:2:2",
    )
    elapsed = time.time() - t0

    assert logits_list[0].shape == (16, cfg.num_classes)

    vram_msg = ""
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() // 1024**2
        vram_msg = f", peak VRAM={peak_mb} MB"

    return f"logits shape={logits_list[0].shape}, {elapsed:.1f}s{vram_msg}"


def test_inference_full_scales(tmp_dir: str):
    """One image, all 6 scales (91 tiles). Use --fast to skip."""
    import torch
    from PIL import Image
    from pipeline.config import PipelineConfig
    from pipeline.model import load_model
    from pipeline.features import get_all_logits

    cfg = PipelineConfig()
    if not Path(cfg.model_path).exists():
        raise _SkipTest("Checkpoint not found — skipping inference test")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_model(cfg.model_name, cfg.num_classes, cfg.model_path)
    img = Image.open(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg").convert("RGB")

    t0 = time.time()
    logits_list = get_all_logits(
        img, "sample_full", cfg.scales, model, tmp_dir,
        tile_size=cfg.tile_size, batch_size=cfg.batch_size,
        use_jpeg_compression=True, jpeg_quality=85, jpeg_subsampling="4:2:2",
    )
    elapsed = time.time() - t0

    total_tiles = sum(arr.shape[0] for arr in logits_list)
    assert total_tiles == 91, f"expected 91 tiles total, got {total_tiles}"

    vram_msg = ""
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() // 1024**2
        vram_msg = f", peak VRAM={peak_mb} MB"

    return f"91 tiles across 6 scales, {elapsed:.1f}s{vram_msg}"


def test_geo_filter():
    from pipeline.config import PipelineConfig
    from pipeline.geo_filter import build_geo_mask

    cfg = PipelineConfig()
    if not Path(cfg.training_metadata_csv).exists():
        raise _SkipTest(f"Metadata CSV not found: {cfg.training_metadata_csv}")

    import numpy as np
    t0 = time.time()
    mask = build_geo_mask(cfg.training_metadata_csv, cfg.num_classes,
                          cfg.geo_ref_lat, cfg.geo_ref_lon, cfg.geo_country_boxes)
    elapsed = time.time() - t0

    assert mask.shape == (cfg.num_classes,) and mask.dtype == bool
    n_kept = int(mask.sum())
    return f"{n_kept}/{cfg.num_classes} species kept ({elapsed:.1f}s)"


def test_prior_computation(tmp_dir: str):
    import numpy as np
    from pipeline.prior import compute_prior_from_logits, save_priors, load_prior_from_file, NUM_CLUSTERS
    from pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    rng = np.random.default_rng(0)
    stems = [
        "2024-CEV3-img001", "LISAH-BOU-img002",  # cluster 0
        "CBN-can-img003", "CBN-PdlC-img004",      # cluster 1
        "CBN-Pla-img005",                          # cluster 2
    ]
    logits = rng.standard_normal((len(stems), cfg.num_classes)).astype(np.float32)

    priors = compute_prior_from_logits(stems, logits)
    assert len(priors) == NUM_CLUSTERS
    for cid, p in priors.items():
        assert p.shape == (cfg.num_classes,)
        assert abs(p.sum() - 1.0) < 1e-3, f"cluster {cid} prior doesn't sum to ~1"

    save_priors(priors, tmp_dir)
    loaded = load_prior_from_file(str(Path(tmp_dir) / "priors.npy"))
    for cid in range(NUM_CLUSTERS):
        assert np.allclose(priors[cid], loaded[cid], atol=1e-6)

    return f"{NUM_CLUSTERS} clusters, save/load round-trip OK"


def test_full_pipeline_run(tmp_dir: str):
    """End-to-end run() call: scale=1, JPEG, prior, no geo (fastest full test)."""
    from pipeline.config import PipelineConfig
    from pipeline.run_pipeline import run
    from pipeline.submission import write_submission

    cfg = PipelineConfig()
    if not Path(cfg.model_path).exists():
        raise _SkipTest("Checkpoint not found — skipping full pipeline test")

    # Copy sample image into a temp images dir
    img_dir = Path(tmp_dir) / "images"
    img_dir.mkdir(exist_ok=True)
    shutil.copy(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg", img_dir)

    cfg.images_dir          = str(img_dir)
    cfg.output_dir          = str(Path(tmp_dir) / "output")
    cfg.cache_dir           = str(Path(tmp_dir) / "cache_pipeline")
    cfg.scales              = [1]
    cfg.use_geo_filter      = False   # skip CSV dependency for this test
    cfg.use_bayesian_prior  = True    # scale=1 is in scales, so prior is computed

    t0 = time.time()
    results = run(cfg)
    elapsed = time.time() - t0

    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    stem = list(results.keys())[0]
    ids  = results[stem]
    assert isinstance(ids, list) and len(ids) > 0

    out_csv = Path(cfg.output_dir) / "submission.csv"
    write_submission(results, str(out_csv))
    assert out_csv.exists()

    return f"1 image → top-{len(ids)} species, {elapsed:.1f}s, CSV written"


def test_full_pipeline_with_geo(tmp_dir: str):
    """Full pipeline with geo filter enabled (requires training metadata CSV)."""
    from pipeline.config import PipelineConfig
    from pipeline.run_pipeline import run

    cfg = PipelineConfig()
    if not Path(cfg.model_path).exists():
        raise _SkipTest("Checkpoint not found")
    if not Path(cfg.training_metadata_csv).exists():
        raise _SkipTest(f"Metadata CSV not found: {cfg.training_metadata_csv}")

    img_dir = Path(tmp_dir) / "images_geo"
    img_dir.mkdir(exist_ok=True)
    shutil.copy(ROOT / "share" / "sample_testimg_2024-CEV3-20240602.jpg", img_dir)

    cfg.images_dir         = str(img_dir)
    cfg.output_dir         = str(Path(tmp_dir) / "output_geo")
    cfg.cache_dir          = str(Path(tmp_dir) / "cache_geo")
    cfg.scales             = [1]
    cfg.use_geo_filter     = True
    cfg.use_bayesian_prior = True

    t0 = time.time()
    results = run(cfg)
    elapsed = time.time() - t0

    ids = list(results.values())[0]
    return f"top-{len(ids)} species after geo filter, {elapsed:.1f}s"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PlantCLEF 2026 HPC smoke-test")
    ap.add_argument("--fast", action="store_true",
                    help="Skip full 6-scale inference test (91 tiles)")
    args = ap.parse_args()

    print("=" * 60)
    print("PlantCLEF 2026 pipeline smoke-test")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="plantclef_test_") as tmp:

        _run("1. imports",               test_imports)
        _run("2. GPU / VRAM",            test_gpu)
        _run("3. tiling geometry",       test_tiling)
        _run("4. JPEG compression",      test_jpeg_compression)
        _run("5. JPEG 4:1:1 (TurboJPEG)", test_jpeg_411)
        _run("6. model load",            test_model_load)
        _run("7. inference scale=1",     test_inference_scale1, tmp)
        _run("8. inference scale=4",     test_inference_scale4, tmp)

        if args.fast:
            _results.append(("9. inference all scales", SKIP, 0.0))
            print(f"[{SKIP}] 9. inference all scales  (--fast flag set)")
        else:
            _run("9. inference all scales", test_inference_full_scales, tmp)

        _run("10. geo filter",           test_geo_filter)
        _run("11. prior compute/save",   test_prior_computation, tmp)
        _run("12. full pipeline run()",  test_full_pipeline_run, tmp)
        _run("13. pipeline + geo filter", test_full_pipeline_with_geo, tmp)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)
    for name, status, secs in _results:
        print(f"  [{status}] {name} ({secs:.1f}s)")

    print()
    print(f"  {n_pass} passed  |  {n_fail} failed  |  {n_skip} skipped")

    try:
        import torch
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() // 1024**2
            print(f"  Peak GPU VRAM across all tests: {peak_mb} MB")
    except Exception:
        pass

    print("=" * 60)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
