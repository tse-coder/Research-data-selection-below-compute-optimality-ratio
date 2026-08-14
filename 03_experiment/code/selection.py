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
New 4th arm (Decision 009):
  5. gradient_norm   - GraNd-style: one forward+backward pass per pool example
                       through the FROZEN Pythia-160m, gradient restricted to
                       the output embedding / LM head only (obtained via
                       model.get_output_embeddings(), not a hardcoded name);
                       score = L2 norm of that one parameter's gradient.
                       Select top-k by score (GraNd's own rule). Backward-pass
                       based, unlike every other method (all forward-only), so
                       a small sanity check is mandatory before the full pool.

Each selection is saved to data/selections/{method}_k{k}_seed{seed}.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _freeze_all_but_output_embeddings(model) -> torch.nn.Module:
    """Freeze every parameter; keep only the output embedding / LM head trainable.

    Uses the HF get_output_embeddings() API (not a hardcoded layer name like
    'embed_out' -- Pythia/GPT-NeoX returns its lm_head Linear there, verified).
    Returns that module.
    """
    for p in model.parameters():
        p.requires_grad = False
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        raise RuntimeError(
            "model.get_output_embeddings() returned None; cannot score by "
            "output-embedding gradient"
        )
    out_emb.weight.requires_grad = True
    return out_emb


@torch.enable_grad()
def gradient_norm_scores(
    pool: list[dict],
    tokenizer,
    model,
    out_emb,
    device,
    limit: int | None = None,
) -> np.ndarray:
    """Per-example GraNd-style score = L2 norm of the output-embedding gradient.

    One forward+backward pass per example over the full instruction+response
    text; gradients accumulate only on the output-embedding weight (all other
    params frozen). Gradient is zeroed between examples.
    """
    model.eval()
    scores = []
    n = len(pool) if limit is None else min(limit, len(pool))
    for i in range(n):
        row = pool[i]
        text = common.format_example(row["instruction"], row.get("input", ""), row.get("output", ""))
        enc = tokenizer(
            text,
            truncation=True,
            max_length=common.MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        grad = out_emb.weight.grad
        scores.append(float(grad.norm().item()))
        model.zero_grad(set_to_none=True)

    return np.asarray(scores, dtype=np.float64)


def gradient_norm_select(
    pool: list[dict], tokenizer, model, k: int, seed: int
) -> np.ndarray:
    """GraNd-style top-k: select the k pool examples with the highest
    output-embedding gradient-norm scores. Deterministic given the scores;
    seed only breaks exact ties (same convention as representation_select)."""
    n = len(pool)
    if k > n:
        raise ValueError(f"k={k} > pool size {n}")
    device = next(model.parameters()).device
    out_emb = _freeze_all_but_output_embeddings(model)
    scores = gradient_norm_scores(pool, tokenizer, model, out_emb, device)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{np.count_nonzero(~np.isfinite(scores))}/{n} "
                           "gradient-norm scores are non-finite")

    order = np.argsort(-scores, kind="stable")
    rng = _rng(seed)
    perm = rng.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[perm] = np.arange(n)
    order_rank = rank[order]
    selected = np.sort(order[:k])
    return selected


def gradient_norm_sanity_check(
    pool: list[dict], tokenizer, model, n: int = 20
) -> None:
    """Mandatory correctness check for the backward-pass code path: score the
    first n examples and print stats. Raises if scores are non-finite or all
    identical; warns on zeros. Call this before scoring the full pool."""
    device = next(model.parameters()).device
    out_emb = _freeze_all_but_output_embeddings(model)
    print(f"get_output_embeddings() -> {type(out_emb).__name__} "
          f"weight shape={tuple(out_emb.weight.shape)}")
    scores = gradient_norm_scores(pool, tokenizer, model, out_emb, device, limit=n)
    print(f"sanity scores (first {n}): {scores}")
    finite = np.all(np.isfinite(scores))
    zero = np.count_nonzero(scores == 0.0)
    print(f"finite={finite} zero={zero}/{n} min={scores.min():.6g} "
          f"max={scores.max():.6g} mean={scores.mean():.6g} "
          f"std={scores.std():.6g}")
    if not finite:
        raise RuntimeError("sanity check FAILED: non-finite gradient-norm scores")
    if np.all(scores == scores[0]):
        raise RuntimeError("sanity check FAILED: all scores identical")
    if zero:
        print(f"WARNING: {zero} scores are exactly zero")
    print("sanity check OK")


def run_selection(method: str, k: int, seed: int) -> dict:
    common.ensure_dirs()
    pool = common.load_pool()
    pool_size = len(pool)
    token_counts = np.asarray(common.load_token_counts())

    tokenizer = None
    model = None
    if method == common.GRADIENT_NORM_METHOD:
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER_NAME)
        model = AutoModelForCausalLM.from_pretrained(common.PROXY_MODEL).to(device)
        model.eval()

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
    elif method == common.GRADIENT_NORM_METHOD:
        indices = gradient_norm_select(pool, tokenizer, model, k, seed)
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
    if method == common.GRADIENT_NORM_METHOD:
        record["selection_start"] = "top-k by output-embedding gradient-norm score (GraNd-style)"
        record["tie_break"] = "seed-ordered permutation on equal scores"
    path = common.SELECTIONS_DIR / f"{method}_k{k}_seed{seed}.json"
    common.save_json(record, path)
    print(json.dumps({k: v for k, v in record.items() if k != "indices"}, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description="Run a data-selection method.")
    parser.add_argument("--method", required=True, choices=common.PRIMARY_METHODS + common.STRETCH_METHODS + [common.GRADIENT_NORM_METHOD])
    parser.add_argument("--k", type=int, default=common.PRIMARY_K)
    parser.add_argument("--seed", type=int, default=common.SPLIT_SEED)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="(gradient_norm only) score the first --sanity-n pool examples and "
             "print stats; do not write a selection file",
    )
    parser.add_argument("--sanity-n", type=int, default=20)
    args = parser.parse_args()

    if args.sanity_check and args.method != common.GRADIENT_NORM_METHOD:
        parser.error("--sanity-check is only valid with --method gradient_norm")

    if args.sanity_check:
        pool = common.load_pool()
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER_NAME)
        model = AutoModelForCausalLM.from_pretrained(common.PROXY_MODEL).to(device)
        model.eval()
        gradient_norm_sanity_check(pool, tokenizer, model, n=args.sanity_n)
        return

    run_selection(args.method, args.k, args.seed)


if __name__ == "__main__":
    main()
