"""Data-selection methods, each timed as its OWN marginal selection runtime
(excluding the shared one-time setup from score_pool.py).

Primary methods (per plan):
  1. random          - uniform random sample of size k from the pool
  2. length_filtered - drop bottom/top 10% by token count, random sample of
                       size k from the remaining middle 80%
  3. representation  - greedy farthest-point (k-center) on L2-normalized
                       final-hidden-layer representations; start at index 0 of
                       the seed-42 shuffle; seed affects tie-breaking only
Stretch method (gated by --method proxy_loss, needs --with-proxy-loss scored):
  4. proxy_loss      - stratify by Pythia-160m mean per-token loss quartiles,
                       sample k/4 uniformly from each quartile

Each selection is saved to data/selections/{method}_k{k}_seed{seed}.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_select(pool_size: int, k: int, seed: int) -> np.ndarray:
    rng = _rng(seed)
    return np.sort(rng.choice(pool_size, size=k, replace=False))


def length_filtered_select(token_counts: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = len(token_counts)
    order = np.argsort(token_counts, kind="stable")
    lo, hi = int(round(0.10 * n)), int(n - round(0.10 * n))
    middle = order[lo:hi]
    if len(middle) < k:
        raise ValueError(f"middle band has {len(middle)} < k={k}")
    rng = _rng(seed)
    return np.sort(rng.choice(middle, size=k, replace=False))


def representation_select(
    reprs: np.ndarray, k: int, seed: int, start_idx: int = 0
) -> np.ndarray:
    n = len(reprs)
    if k > n:
        raise ValueError(f"k={k} > pool size {n}")
    rng = _rng(seed)
    perm = rng.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[perm] = np.arange(n)

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


def proxy_loss_select(losses: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k % 4 != 0:
        raise ValueError(f"k={k} not divisible by 4 for proxy-loss stratification")
    per_quartile = k // 4
    quartiles = np.quantile(losses, [0.25, 0.5, 0.75])
    bins = np.digitize(losses, quartiles)
    rng = _rng(seed)
    out = []
    for q in range(4):
        members = np.flatnonzero(bins == q)
        if len(members) < per_quartile:
            raise ValueError(f"quartile {q} has {len(members)} < {per_quartile} needed")
        out.append(rng.choice(members, size=per_quartile, replace=False))
    return np.sort(np.concatenate(out))


def run_selection(method: str, k: int, seed: int) -> dict:
    common.ensure_dirs()
    pool = common.load_pool()
    pool_size = len(pool)
    token_counts = np.asarray(common.load_token_counts())

    t0 = time.perf_counter()
    if method == "random":
        indices = random_select(pool_size, k, seed)
    elif method == "length_filtered":
        indices = length_filtered_select(token_counts, k, seed)
    elif method == "representation":
        repr_path = common.DATA_DIR / "pool_reprs.npy"
        if not repr_path.exists():
            raise FileNotFoundError(
                f"{repr_path} missing; run score_pool.py first (shared one-time setup)"
            )
        reprs = np.load(repr_path)
        indices = representation_select(reprs, k, seed)
    elif method == "proxy_loss":
        loss_path = common.DATA_DIR / "pool_proxy_losses.npy"
        if not loss_path.exists():
            raise FileNotFoundError(
                f"{loss_path} missing; run score_pool.py --with-proxy-loss first (stretch method)"
            )
        losses = np.load(loss_path)
        indices = proxy_loss_select(losses, k, seed)
    else:
        raise ValueError(f"unknown method: {method}")
    runtime_s = time.perf_counter() - t0

    selected_tokens_raw = int(token_counts[indices].sum())
    selected_tokens_effective = int(np.minimum(token_counts[indices], common.MAX_SEQ_LEN).sum())

    record = {
        "method": method,
        "k": int(k),
        "seed": int(seed),
        "pool_size": pool_size,
        "indices": indices.tolist(),
        "selection_runtime_s": runtime_s,
        "selected_example_count": int(len(indices)),
        "selected_token_count_raw": selected_tokens_raw,
        "selected_token_count_effective": selected_tokens_effective,
        "mean_tokens_per_example_raw": float(token_counts[indices].mean()),
        "selection_start": "index 0 of seed-42 shuffle (representation only)",
        "tie_break": "seed-ordered permutation (representation only)",
    }
    path = common.SELECTIONS_DIR / f"{method}_k{k}_seed{seed}.json"
    common.save_json(record, path)
    print(json.dumps({k: v for k, v in record.items() if k != "indices"}, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description="Run a data-selection method.")
    parser.add_argument("--method", required=True, choices=common.PRIMARY_METHODS + common.STRETCH_METHODS)
    parser.add_argument("--k", type=int, default=common.PRIMARY_K)
    parser.add_argument("--seed", type=int, default=common.SPLIT_SEED)
    args = parser.parse_args()
    run_selection(args.method, args.k, args.seed)


if __name__ == "__main__":
    main()
