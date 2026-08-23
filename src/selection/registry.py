"""Selection registry and the run_selection driver (timing + record schema).

Heavy dependencies are imported lazily per method so that, e.g., listing or
running `random` does not require torch.
"""
from __future__ import annotations

import json
import time

import numpy as np

from src import config

ALL_SELECTORS = [
    "random",
    "length_filtered",
    "representation",
    "proxy_loss",
    "gradient_norm",
]


def _selectors():
    from src.selection import (
        gradient_norm_selector,
        length_filtered_selector,
        random_selector,
        representation_selector,
        proxy_loss_selector,
    )

    return {
        "random": random_selector.select,
        "length_filtered": length_filtered_selector.select,
        "representation": representation_selector.select,
        "proxy_loss": proxy_loss_selector.select,
        "gradient_norm": gradient_norm_selector.select,
    }


def run_selection(method: str, k: int, seed: int, save_diagnostics: bool = False) -> dict:
    config.ensure_dirs()
    pool = config.load_pool()
    pool_size = len(pool)
    token_counts = np.asarray(config.load_token_counts())

    tokenizer = None
    model = None
    if method == config.GRADIENT_NORM_METHOD:
        from src.selection.gradient_norm_selector import load_scorer

        tokenizer, model, _ = load_scorer()

    t0 = time.perf_counter()
    if method == "random":
        indices = _selectors()["random"](pool_size, k, seed)
    elif method == "length_filtered":
        indices = _selectors()["length_filtered"](token_counts, k, seed)
    elif method == "representation":
        repr_path = config.DATA_DIR / "pool_reprs.npy"
        if not repr_path.exists():
            raise FileNotFoundError(
                f"{repr_path} missing; run `python -m src score` first (shared one-time setup)"
            )
        reprs = np.load(repr_path)
        indices = _selectors()["representation"](reprs, k, seed)
    elif method == config.GRADIENT_NORM_METHOD:
        diag_path = None
        if save_diagnostics:
            diag_path = config.DATA_DIR / "gradient_norm_diagnostics.json"
        indices = _selectors()[config.GRADIENT_NORM_METHOD](
            pool, tokenizer, model, k, seed, diagnostics_path=diag_path
        )
    elif method == "proxy_loss":
        loss_path = config.DATA_DIR / "pool_proxy_losses.npy"
        if not loss_path.exists():
            raise FileNotFoundError(
                f"{loss_path} missing; run `python -m src score --with-proxy-loss` first"
            )
        losses = np.load(loss_path)
        indices = _selectors()["proxy_loss"](losses, k, seed)
    else:
        raise ValueError(f"unknown method: {method}")
    runtime_s = time.perf_counter() - t0

    selected_tokens_raw = int(token_counts[indices].sum())
    selected_tokens_effective = int(np.minimum(token_counts[indices], config.MAX_SEQ_LEN).sum())

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
    if method == config.GRADIENT_NORM_METHOD:
        record["selection_start"] = "top-k by output-embedding gradient-norm score (GraNd-style)"
        record["tie_break"] = "seed-ordered permutation on equal scores"
    path = config.SELECTIONS_DIR / f"{method}_k{k}_seed{seed}.json"
    config.save_json(record, path)
    print(json.dumps({k: v for k, v in record.items() if k != "indices"}, indent=2))
    return record


def gradient_norm_sanity_check(n: int = 20) -> None:
    from src.selection.gradient_norm_selector import sanity_check

    pool = config.load_pool()
    _, model, _ = __import__(
        "src.selection.gradient_norm_selector", fromlist=["load_scorer"]
    ).load_scorer()
    sanity_check(pool, model=model, n=n)


if __name__ == "__main__":
    raise SystemExit("Use `python -m src select ...` (see cli.py)")
