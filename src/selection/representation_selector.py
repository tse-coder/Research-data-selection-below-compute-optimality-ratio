"""Representation-diversity selection: greedy farthest-point (k-center) over
cached L2-normalized final-hidden-layer vectors. Vectors live on the unit
sphere, so dot product equals cosine similarity and distance is 1 - dot.
Starts at index 0 of the seed-42 shuffle; seed affects tie-breaking only."""
from __future__ import annotations

import numpy as np

from src.selection.base import tie_break_rank


def select(reprs: np.ndarray, k: int, seed: int, start_idx: int = 0) -> np.ndarray:
    n = len(reprs)
    if k > n:
        raise ValueError(f"k={k} > pool size {n}")
    rank = tie_break_rank(n, seed)

    selected = [start_idx]
    min_dist = 1.0 - reprs @ reprs[start_idx]
    min_dist[start_idx] = -np.inf

    for _ in range(k - 1):
        max_val = min_dist.max()
        candidates = np.flatnonzero(np.isclose(min_dist, max_val, atol=1e-12))
        nxt = candidates[np.argmin(rank[candidates])]
        selected.append(int(nxt))
        new_dist = 1.0 - reprs @ reprs[nxt]
        np.minimum(min_dist, new_dist, out=min_dist)
        min_dist[nxt] = -np.inf

    return np.sort(np.asarray(selected))
