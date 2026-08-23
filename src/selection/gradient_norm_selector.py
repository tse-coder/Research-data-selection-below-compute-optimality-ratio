"""Gradient-norm selection (GraNd-style, Paul et al. 2021).

Score = L2 norm of the output-embedding / LM-head gradient when the frozen
Pythia-160m scorer reads each pool example (one forward+backward pass per
example). The head is obtained via model.get_output_embeddings() — never a
hardcoded layer name, since architecture-specific names caused a real bug
earlier in this project (q_proj/v_proj vs query_key_value). Top-k by score,
GraNd's own rule; deterministic given the scores, seed breaks exact ties.

Backward-pass based, unlike every other method (all forward-only), so a small
sanity check is mandatory before scoring the full pool. --diagnostics saves
per-example length/score/top-k data and prints the score~length summary.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config
from src.selection.base import spearman_corr, tie_break_rank


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_scorer():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(config.PROXY_MODEL).to(device)
    model.eval()
    return tokenizer, model, device


def freeze_all_but_output_embeddings(model) -> torch.nn.Module:
    """Freeze every parameter; keep only the output embedding / LM head trainable."""
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
    """Per-example GraNd-style score; gradients zeroed between examples."""
    model.eval()
    scores = []
    n = len(pool) if limit is None else min(limit, len(pool))
    for i in range(n):
        row = pool[i]
        text = config.format_example(row["instruction"], row.get("input", ""), row.get("output", ""))
        enc = tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
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


def select(
    pool: list[dict], tokenizer, model, k: int, seed: int, diagnostics_path=None
) -> np.ndarray:
    n = len(pool)
    if k > n:
        raise ValueError(f"k={k} > pool size {n}")
    device = next(model.parameters()).device
    out_emb = freeze_all_but_output_embeddings(model)
    scores = gradient_norm_scores(pool, tokenizer, model, out_emb, device)
    _require_finite(scores, n)

    order = np.argsort(-scores, kind="stable")
    rank = tie_break_rank(n, seed)
    order_rank = rank[order]
    selected = np.sort(order[:k])

    if diagnostics_path is not None:
        _write_diagnostics(diagnostics_path, scores, selected, k, seed, n)
    return selected


def _require_finite(scores: np.ndarray, n: int) -> None:
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            f"{np.count_nonzero(~np.isfinite(scores))}/{n} gradient-norm scores are non-finite"
        )


def sanity_check(pool: list[dict], tokenizer, model, n: int = 20) -> None:
    """Mandatory correctness check for the backward-pass code path."""
    device = next(model.parameters()).device
    out_emb = freeze_all_but_output_embeddings(model)
    print(f"get_output_embeddings() -> {type(out_emb).__name__} "
          f"weight shape={tuple(out_emb.weight.shape)}")
    scores = gradient_norm_scores(pool, tokenizer, model, out_emb, device, limit=n)
    print(f"sanity scores (first {n}): {scores}")
    finite = np.all(np.isfinite(scores))
    zero = np.count_nonzero(scores == 0.0)
    print(f"finite={finite} zero={zero}/{n} min={scores.min():.6g} "
          f"max={scores.max():.6g} mean={scores.mean():.6g} std={scores.std():.6g}")
    if not finite:
        raise RuntimeError("sanity check FAILED: non-finite gradient-norm scores")
    if np.all(scores == scores[0]):
        raise RuntimeError("sanity check FAILED: all scores identical")
    if zero:
        print(f"WARNING: {zero} scores are exactly zero")
    print("sanity check OK")


def _write_diagnostics(path, scores, selected, k, seed, n) -> None:
    token_counts = np.asarray(config.load_token_counts(), dtype=int)
    if len(token_counts) != n:
        raise ValueError(f"token-count file has {len(token_counts)} entries, pool has {n}")
    in_selected = np.zeros(n, dtype=bool)
    in_selected[selected] = True
    sel_len, rest_len = token_counts[in_selected], token_counts[~in_selected]
    diag = {
        "method": config.GRADIENT_NORM_METHOD,
        "k": int(k),
        "seed": int(seed),
        "pool_size": int(n),
        "score_definition": (
            "L2 norm of output-embedding gradient (frozen Pythia-160m, "
            "one fwd+bwd per example, GraNd-style)"
        ),
        "token_count_definition": "raw untruncated Pythia-token count (same as all other tables)",
        "examples": [
            {
                "pool_index": int(i),
                "token_count": int(token_counts[i]),
                "gradient_norm_score": float(scores[i]),
                "selected": bool(in_selected[i]),
            }
            for i in range(n)
        ],
    }
    config.save_json(diag, path)
    rho = spearman_corr(scores, token_counts)
    print(f"Saved diagnostics to {path}")
    print("token lengths SELECTED top-%d: mean=%.1f median=%.1f min=%d max=%d"
          % (len(sel_len), sel_len.mean(), np.median(sel_len), sel_len.min(), sel_len.max()))
    print("token lengths REST (%d): mean=%.1f median=%.1f min=%d max=%d"
          % (len(rest_len), rest_len.mean(), np.median(rest_len), rest_len.min(), rest_len.max()))
    print(f"Spearman(gradient_norm_score, token_length) over all {n}: {rho:.4f}")
