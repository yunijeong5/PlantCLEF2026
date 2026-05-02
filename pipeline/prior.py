"""
Cluster-based Bayesian prior (paper 2 post-processing).

Cluster assignment
------------------
Each test image is assigned to one of three clusters derived in paper 2 by
running PaCMAP + KMeans on DINOv2 CLS-token embeddings of the test set.  The
dominant cluster for each geographic region is hardcoded from that analysis:

  Cluster 0 — majority of regions (LISAH, GUARDEN, OPTMix, RNNB, CBN-Pyr, …)
  Cluster 1 — CBN-PdlC, CBN-can
  Cluster 2 — CBN-Pla

Assignment is done purely by matching the image filename against known region
prefixes, so no embedding step is needed at test time.

Prior computation
-----------------
P(y | cluster c) is the element-wise average of softmax(logits) over all
test images that belong to cluster c.  When scale-1 logits are already
cached (they are always computed as part of the multi-scale run), the prior
is computed automatically.  You can also supply a pre-computed file via
PipelineConfig.prior_data_path.

File format for prior_data_path
--------------------------------
A .npy file containing a float32 array of shape [NUM_CLUSTERS, num_classes].
Row index equals cluster_id (row 0 = cluster 0, row 1 = cluster 1, …).
This is also the format written by save_priors().

Application
-----------
Image-level probabilities p (shape [num_classes]) are multiplied element-wise:
  p_weighted = p * prior[cluster_id]

Because prior is the same for all tiles, this commutes with every aggregation
method; we always apply it AFTER aggregation (see aggregation.py).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.special import softmax  # numerically stable

# ── Cluster prefix mapping (from paper 2 analysis) ───────────────────────────

REGION_PREFIXES: List[str] = [
    "2024-CEV3",
    "CBN-can",
    "CBN-PdlC",
    "CBN-Pla",
    "CBN-Pyr",
    "GUARDEN-AMB",
    "GUARDEN-CBNMed",
    "LISAH-BOU",
    "LISAH-BVD",
    "LISAH-JAS",
    "LISAH-PEC",
    "OPTMix",
    "RNNB",
]

# Dominant cluster per region prefix (paper 2, clustering notebook)
REGION_TO_CLUSTER: Dict[str, int] = {
    "GUARDEN-CBNMed": 0,
    "RNNB":           0,
    "LISAH-BOU":      0,
    "OPTMix":         0,
    "LISAH-BVD":      0,
    "GUARDEN-AMB":    0,
    "LISAH-PEC":      0,
    "LISAH-JAS":      0,
    "CBN-Pyr":        0,
    "2024-CEV3":      0,
    "CBN-PdlC":       1,
    "CBN-can":        1,
    "CBN-Pla":        2,
}

NUM_CLUSTERS = 3
DEFAULT_CLUSTER = 0   # fallback for unrecognised prefixes

# Build a single regex that matches any known prefix at the start of the name
_PREFIX_PATTERN = re.compile(
    "^(" + "|".join(re.escape(p) for p in sorted(REGION_PREFIXES, key=len, reverse=True)) + ")"
)


def get_cluster(image_name: str) -> int:
    """
    Return the cluster id (0, 1, or 2) for an image given its filename stem.
    Matches the longest known region prefix at the start of the name.
    """
    m = _PREFIX_PATTERN.match(image_name)
    if m:
        return REGION_TO_CLUSTER.get(m.group(1), DEFAULT_CLUSTER)
    return DEFAULT_CLUSTER


# ── Prior loading / computation ───────────────────────────────────────────────

def load_prior_from_file(path: str) -> Dict[int, np.ndarray]:
    """
    Load cluster priors from a .npy file.
    Expected shape: [NUM_CLUSTERS, num_classes], float32.
    Row index equals cluster_id.
    Returns {cluster_id: np.array([num_classes], float32)}.
    """
    arr = np.load(path)   # [NUM_CLUSTERS, num_classes]
    return {cid: arr[cid] for cid in range(len(arr))}


def compute_prior_from_logits(
    stems: List[str],
    scale1_logits: np.ndarray,   # [N, num_classes]
) -> Dict[int, np.ndarray]:
    """
    Derive P(y | cluster) by averaging softmax probabilities within each cluster.

    *scale1_logits* should be the raw model logits for each image at scale 1
    (one 518×518 whole-image crop), which are available from the cache after
    running the multi-scale pipeline.
    """
    num_classes = scale1_logits.shape[1]
    cluster_sums   = np.zeros((NUM_CLUSTERS, num_classes), dtype=np.float64)
    cluster_counts = np.zeros(NUM_CLUSTERS, dtype=np.int64)

    probs = softmax(scale1_logits, axis=1).astype(np.float64)  # [N, C]

    for i, stem in enumerate(stems):
        cid = get_cluster(stem)
        cluster_sums[cid]   += probs[i]
        cluster_counts[cid] += 1

    priors: Dict[int, np.ndarray] = {}
    for cid in range(NUM_CLUSTERS):
        if cluster_counts[cid] > 0:
            priors[cid] = (cluster_sums[cid] / cluster_counts[cid]).astype(np.float32)
        else:
            # Flat (uninformative) prior for empty clusters
            priors[cid] = np.ones(num_classes, dtype=np.float32) / num_classes

    return priors


def save_priors(priors: Dict[int, np.ndarray], cache_dir: str) -> None:
    """
    Persist computed priors as a .npy file of shape [NUM_CLUSTERS, num_classes].
    Row index equals cluster_id.
    """
    arr = np.stack([priors[cid] for cid in range(NUM_CLUSTERS)])  # [3, num_classes]
    out = Path(cache_dir) / "priors.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)


def apply_prior(
    image_probs: np.ndarray,        # [num_classes]
    image_name: str,
    priors: Dict[int, np.ndarray],  # {cluster_id: [num_classes]}
    strength: float = 1.0,
) -> np.ndarray:
    """
    Multiply image-level probabilities by the cluster prior.

    *strength* controls how strongly the prior affects ranking:
      1.0 -> image_probs * prior
      0.5 -> image_probs * sqrt(prior)
      0.0 -> image_probs

    The result is NOT re-normalised so the relative ranking is simply rescaled.
    """
    if strength < 0.0:
        raise ValueError("prior strength must be non-negative.")
    if strength == 0.0:
        return image_probs

    cid = get_cluster(image_name)
    prior = priors.get(cid, priors.get(DEFAULT_CLUSTER))
    return image_probs * np.power(prior, strength)
