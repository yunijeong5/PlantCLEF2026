"""
Geographical species filter (paper 2 post-processing).

Method (derived from the PlantCLEF2025 test-set geography):
  1. Load training metadata (PlantCLEF2024_single_plant_training_metadata.csv).
     Columns used: species_id, latitude, longitude.
  2. For each species, find the observation whose coordinates are closest to
     the reference point (default: 44°N, 4°E — Southern France) using squared
     Euclidean distance in (lat, lon) space.
  3. Keep species whose nearest observation falls within the bounding box of
     at least one target country (France / Spain / Italy / Switzerland).

The result — a boolean numpy mask of shape [num_classes] — is cached to disk
so the expensive CSV scan only runs once.

Design note:
  Squared Euclidean distance in lat/lon is a rough but fast approximation
  (1° lat ≈ 111 km, 1° lon ≈ 111·cos(lat) km at mid-latitudes).  For a
  regional filter this is sufficient; swapping to Haversine distance would
  not materially change which species are retained.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _in_any_box(lat: float, lon: float, boxes: List[List[float]]) -> bool:
    """Return True if (lat, lon) falls inside at least one bounding box.
    Each box is [lat_min, lat_max, lon_min, lon_max].
    """
    for lat_min, lat_max, lon_min, lon_max in boxes:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


def build_geo_mask(
    metadata_csv: str,
    num_classes: int,
    ref_lat: float = 44.0,
    ref_lon: float = 4.0,
    country_boxes: List[List[float]] = None,
) -> np.ndarray:
    """
    Compute a boolean mask of shape [num_classes] where True means the species
    has at least one training observation near the target region.

    Species whose species_id is not present in the training metadata are kept
    (mask=True) to avoid silently discarding novel species.
    """
    if country_boxes is None:
        country_boxes = [
            [41.3, 51.1, -5.2,  9.6],   # France
            [35.9, 43.8, -9.3,  4.3],   # Spain
            [35.5, 47.1,  6.6, 18.5],   # Italy
            [45.8, 47.8,  5.9, 10.5],   # Switzerland
        ]

    df = pd.read_csv(metadata_csv, sep=";", usecols=["species_id", "latitude", "longitude"])
    df = df.dropna(subset=["latitude", "longitude"])
    df["species_id"] = df["species_id"].astype(int)

    # Squared distance from reference point (no sqrt needed — only need argmin)
    df["dist2"] = (df["latitude"] - ref_lat) ** 2 + (df["longitude"] - ref_lon) ** 2

    # For each species, pick the observation closest to the reference point
    nearest = df.loc[df.groupby("species_id")["dist2"].idxmin()]

    # Check which nearest observations fall inside a target country
    in_region = nearest.apply(
        lambda row: _in_any_box(row["latitude"], row["longitude"], country_boxes),
        axis=1,
    )
    valid_ids = set(nearest.loc[in_region, "species_id"].tolist())

    # Build the mask; default True for species absent from metadata
    mask = np.ones(num_classes, dtype=bool)
    all_ids_in_metadata = set(nearest["species_id"].tolist())
    for sid in all_ids_in_metadata:
        if sid < num_classes:
            mask[sid] = sid in valid_ids

    return mask


def load_or_build_geo_mask(
    metadata_csv: str,
    num_classes: int,
    cache_dir: str,
    ref_lat: float = 44.0,
    ref_lon: float = 4.0,
    country_boxes: List[List[float]] = None,
) -> np.ndarray:
    """Load the geo mask from cache, or compute and cache it."""
    cache_path = Path(cache_dir) / "geo_mask.npy"
    if cache_path.exists():
        return np.load(cache_path)

    print("Building geo mask from training metadata (runs once)...")
    mask = build_geo_mask(metadata_csv, num_classes, ref_lat, ref_lon, country_boxes)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mask)
    n_valid = int(mask.sum())
    print(f"Geo mask built: {n_valid}/{num_classes} species retained.")
    return mask


def apply_geo_filter(probs: np.ndarray, geo_mask: np.ndarray) -> np.ndarray:
    """Zero out species not present in the target region."""
    result = probs.copy()
    result[~geo_mask] = 0.0
    return result
