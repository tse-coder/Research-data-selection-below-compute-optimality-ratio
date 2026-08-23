"""Length-filtered selection: drop the bottom/top 10% by token count, then
uniformly sample k from the remaining middle 80%."""
from __future__ import annotations

import numpy as np

from src.selection.base import rng


def select(token_counts: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = len(token_counts)
    order = np.argsort(token_counts, kind="stable")
    lo, hi = int(round(0.10 * n)), int(n - round(0.10 * n))
    middle = order[lo:hi]
    if len(middle) < k:
        raise ValueError(f"middle band has {len(middle)} < k={k}")
    r = rng(seed)
    return np.sort(r.choice(middle, size=k, replace=False))
