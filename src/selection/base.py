"""Shared helpers for selection methods: seeding, ranking, diagnostics math."""
from __future__ import annotations

import numpy as np


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def tie_break_rank(n: int, seed: int) -> np.ndarray:
    """Seed-ordered permutation rank: lower rank wins exact ties."""
    r = rng(seed)
    perm = r.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[perm] = np.arange(n)
    return rank


def average_ranks(a: np.ndarray) -> np.ndarray:
    """1-based average ranks with ties averaged (Spearman-correct)."""
    a = np.asarray(a)
    order = np.argsort(a, kind="stable")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_corr(x, y) -> float:
    """Spearman rank correlation (tie-corrected), no scipy dependency."""
    rx = average_ranks(np.asarray(x, dtype=np.float64))
    ry = average_ranks(np.asarray(y, dtype=np.float64))
    return float(np.corrcoef(rx, ry)[0, 1])
