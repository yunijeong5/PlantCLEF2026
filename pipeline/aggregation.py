"""
Tile aggregation strategies.

Each function receives *tile_probs* of shape [T, num_classes] — the softmax
probabilities for all T tiles of one image — and returns a 1-D array of
shape [num_classes] representing the image-level prediction.

Supported methods (selectable via PipelineConfig.aggregation):

  "max"       — maximum per species across all tiles (paper 1 default).
                Captures the most salient tile for each species.

  "mean"      — arithmetic mean across all tiles.
                More robust to spurious high-confidence tiles.

  "topk_mean" — average of the top-k tile values per species.
                A smooth interpolation between max and mean; k is set by
                PipelineConfig.topk_mean_k.

Note on prior application order:
  Since P(y|cluster) is constant across all tiles of one image,
  multiplying by the prior commutes with all three aggregation methods:
    agg_method(p_t * prior) == agg_method(p_t) * prior
  We therefore apply the prior AFTER aggregation for efficiency.
"""

import numpy as np


def aggregate_max(tile_probs: np.ndarray) -> np.ndarray:
    """Shape [T, C] → [C], element-wise maximum over tiles."""
    return tile_probs.max(axis=0)


def aggregate_mean(tile_probs: np.ndarray) -> np.ndarray:
    """Shape [T, C] → [C], arithmetic mean over tiles."""
    return tile_probs.mean(axis=0)


def aggregate_topk_mean(tile_probs: np.ndarray, k: int) -> np.ndarray:
    """
    Shape [T, C] → [C].
    For each species, average the k highest tile probabilities.
    When k >= T this is equivalent to the plain mean.
    """
    k = min(k, tile_probs.shape[0])
    # Partial sort: get top-k along tile axis without full sort
    top_k = np.partition(tile_probs, -k, axis=0)[-k:]   # [k, C]
    return top_k.mean(axis=0)


def aggregate(
    tile_logits: np.ndarray,
    method: str = "max",
    topk_mean_k: int = 5,
) -> np.ndarray:
    """
    Convert raw logits [T, C] to an aggregated image-level probability [C].

    Softmax is applied before aggregation so that each tile's probability
    distribution sums to 1 (matching paper 2's per-tile softmax convention).
    """
    # Numerically stable softmax along the class axis
    shifted = tile_logits - tile_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    tile_probs = exp / exp.sum(axis=1, keepdims=True)   # [T, C]

    if method == "max":
        return aggregate_max(tile_probs)
    if method == "mean":
        return aggregate_mean(tile_probs)
    if method == "topk_mean":
        return aggregate_topk_mean(tile_probs, topk_mean_k)

    raise ValueError(f"Unknown aggregation method: {method!r}. "
                     "Choose 'max', 'mean', or 'topk_mean'.")
