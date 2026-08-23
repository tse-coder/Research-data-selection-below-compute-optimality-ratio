"""Random selection: seeded uniform sample of k pool indices (the baseline)."""
from __future__ import annotations

import numpy as np

from src.selection.base import rng


def select(pool_size: int, k: int, seed: int) -> np.ndarray:
    r = rng(seed)
    return np.sort(r.choice(pool_size, size=k, replace=False))
