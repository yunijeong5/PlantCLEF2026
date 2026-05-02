from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PipelineConfig:
    # ── I/O ──────────────────────────────────────────────────────────────────
    images_dir: str = str(PROJECT_ROOT / "data" / "test" / "images")
    output_dir: str = str(PROJECT_ROOT / "output")
    # Cache root; features and logits land in sub-directories of this path.
    # Use different values to keep caches for different pre-processing configs
    # (e.g. "cache_jpeg85" vs "cache_nojpeg").
    cache_dir: str = str(PROJECT_ROOT / "cache")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    model_path: str = str(
        PROJECT_ROOT
        / "pretrained_models"
        / "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
        / "model_best.pth.tar"
    )
    class_mapping_file: str = str(PROJECT_ROOT / "pretrained_models" / "class_mapping.txt")
    num_classes: int = 7806
    tile_size: int = 518
    # Number of tiles to forward in one GPU batch across all scales.
    batch_size: int = 32

    # ── Pre-processing — paper 1 ──────────────────────────────────────────────
    # Whether to apply a JPEG round-trip on every 518×518 tile before inference.
    # This aligns test-domain compression artifacts with the training data.
    # The round-trip is necessary even for .jpg test images because their
    # original encoding quality and subsampling are unknown.
    use_jpeg_compression: bool = True
    jpeg_quality: int = 85
    # Chroma subsampling mode for the JPEG round-trip.
    # Paper 1 found two equivalent best configurations:
    #   "4:2:2" at quality 85  — Pillow native, default here
    #   "4:1:1" at quality 94  — requires:  pip install PyTurboJPEG
    # Also accepted: "4:4:4" (no subsampling), "4:2:0" (Pillow native).
    # NOTE: Pillow's subsampling=2 produces 4:2:0, not 4:1:1.
    jpeg_subsampling: str = "4:2:2"

    # Tiling scales. Each entry S produces S² non-overlapping 518×518 tiles.
    # [6, 5, 4, 3, 2, 1] → 91 tiles total (paper 1 full multi-scale).
    # [4] → 16 tiles (paper 2 single-scale baseline).
    # [1] → 1 tile (no tiling baseline).
    scales: List[int] = field(default_factory=lambda: [6, 5, 4, 3, 2, 1])

    # ── Aggregation ───────────────────────────────────────────────────────────
    # How to combine probabilities across all tiles of an image.
    #   "max"       — take the maximum per species (paper 1 default)
    #   "mean"      — average across all tiles
    #   "topk_mean" — average of the top-k tile values per species (use topk_mean_k)
    aggregation: Literal["max", "mean", "topk_mean"] = "max"
    topk_mean_k: int = 5  # only used when aggregation == "topk_mean"

    # ── Post-processing — paper 2 ─────────────────────────────────────────────
    # Bayesian prior: multiply image-level probabilities by P(y | cluster).
    # If prior_data_path is None and use_bayesian_prior is True, the prior is
    # computed on-the-fly from scale-1 predictions cached in cache_dir.
    use_bayesian_prior: bool = True
    # Path to a parquet/CSV with columns [cluster_id (int), prior_probabilities (list[float])].
    # Leave None to compute from the scale-1 tile cache.
    prior_data_path: Optional[str] = None

    # Geographical filter: zero out species with no known observation in the
    # target region.  The valid species set is derived from training metadata.
    use_geo_filter: bool = True
    training_metadata_csv: str = str(
        PROJECT_ROOT
        / "data"
        / "train_singleplant"
        / "PlantCLEF2024_single_plant_training_metadata.csv"
    )
    # Reference point for the test set: Southern France.
    geo_ref_lat: float = 44.0
    geo_ref_lon: float = 4.0
    # Approximate bounding boxes of target countries [lat_min, lat_max, lon_min, lon_max].
    geo_country_boxes: List[List[float]] = field(
        default_factory=lambda: [
            [41.3, 51.1, -5.2, 9.6],   # France
            [35.9, 43.8, -9.3, 4.3],   # Spain
            [35.5, 47.1, 6.6, 18.5],   # Italy
            [45.8, 47.8, 5.9, 10.5],   # Switzerland
        ]
    )

    # ── Submission ────────────────────────────────────────────────────────────
    top_k: int = 15
    min_score: float = 0.01
