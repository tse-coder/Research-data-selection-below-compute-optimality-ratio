"""Evaluation utilities: per-example eval losses and bootstrap confidence.

Eval loss scope (fixed): full-sequence mean cross-entropy over non-padding
tokens, computed on the fixed 800-example held-out split.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src import config


@torch.no_grad()
def per_example_eval_losses(
    model, tokenizer, texts, device, batch_size=config.EVAL_BATCH_SIZE, max_len=config.MAX_SEQ_LEN
) -> np.ndarray:
    model.eval()
    losses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        labels[enc["attention_mask"] == 0] = -100
        logits = model(**enc).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != -100).float()
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )
        ce = ce.reshape(shift_labels.shape)
        loss = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        losses.append(loss.float().cpu().numpy())
    return np.concatenate(losses, axis=0)


def bootstrap_ci(
    losses: np.ndarray, n=config.BOOTSTRAP_N, seed=config.BOOTSTRAP_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    n_ex = len(losses)
    samples = rng.choice(losses, size=(n, n_ex), replace=True).mean(axis=1)
    return {
        "n_resamples": n,
        "seed": seed,
        "mean": float(losses.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "per_example_losses": losses.tolist(),
    }
