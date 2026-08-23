"""Proxy-loss selection (stretch method): stratified sampling over cached
Pythia-160m mean per-token loss quartiles, k/4 per quartile."""
from __future__ import annotations

import numpy as np

from src.selection.base import rng


def select(losses: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k % 4 != 0:
        raise ValueError(f"k={k} not divisible by 4 for proxy-loss stratification")
    per_quartile = k // 4
    quartiles = np.quantile(losses, [0.25, 0.5, 0.75])
    bins = np.digitize(losses, quartiles)
    r = rng(seed)
    out = []
    for q in range(4):
        members = np.flatnonzero(bins == q)
        if len(members) < per_quartile:
            raise ValueError(f"quartile {q} has {len(members)} < {per_quartile} needed")
        out.append(r.choice(members, size=per_quartile, replace=False))
    return np.sort(np.concatenate(out))
