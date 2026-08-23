"""One-time shared setup: frozen Pythia-160m extraction over the pool.

Extracts final-hidden-layer, attention-mask-aware mean-pooled, L2-normalized
representations for every pool example (data/pool_reprs.npy), optionally also
mean per-token proxy losses (--with-proxy-loss, stretch method only).

The extraction runtime is the shared one-time setup cost and is NOT charged
per selection method.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_texts(pool: list[dict]) -> list[str]:
    return [
        config.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        for r in pool
    ]


@torch.no_grad()
def extract_reprs(
    model, tokenizer, texts, device, batch_size=config.SCORE_BATCH_SIZE, max_len=config.MAX_SEQ_LEN
) -> np.ndarray:
    model.eval()
    reprs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        hidden = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=-1)
        reprs.append(pooled.float().cpu().numpy())
    return np.concatenate(reprs, axis=0)


@torch.no_grad()
def compute_proxy_losses(
    model, tokenizer, texts, device, batch_size=config.SCORE_BATCH_SIZE, max_len=config.MAX_SEQ_LEN
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


def run(force: bool = False, with_proxy_loss: bool = False) -> None:
    config.ensure_dirs()
    repr_path = config.DATA_DIR / "pool_reprs.npy"
    loss_path = config.DATA_DIR / "pool_proxy_losses.npy"
    if not force and repr_path.exists() and (not with_proxy_loss or loss_path.exists()):
        print("Scoring output already present; use --force to recompute.")
        return

    pool = config.load_pool()
    texts = load_texts(pool)
    device = get_device()
    print(f"Device: {device}")

    t_model = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.PROXY_MODEL, output_hidden_states=True
    ).to(device)
    model.eval()
    model_load_s = time.perf_counter() - t_model

    t_extract = time.perf_counter()
    reprs = extract_reprs(model, tokenizer, texts, device)
    extract_s = time.perf_counter() - t_extract

    np.save(repr_path, reprs.astype(np.float32))
    print(f"Saved {repr_path} shape={reprs.shape}")

    if with_proxy_loss:
        t_loss = time.perf_counter()
        losses = compute_proxy_losses(model, tokenizer, texts, device)
        loss_s = time.perf_counter() - t_loss
        np.save(loss_path, losses.astype(np.float32))
        print(f"Saved {loss_path} shape={losses.shape} proxy-loss compute {loss_s:.1f}s")

    meta = {
        "model": config.PROXY_MODEL,
        "pool_size": len(pool),
        "batch_size": config.SCORE_BATCH_SIZE,
        "max_seq_len": config.MAX_SEQ_LEN,
        "tokenizer": config.TOKENIZER_NAME,
        "device": str(device),
        "pooling": "final hidden layer, attention-mask-aware mean, L2-normalized",
        "model_load_s": model_load_s,
        "extract_runtime_s": extract_s,
        "with_proxy_loss": with_proxy_loss,
    }
    config.save_json(meta, config.DATA_DIR / "scoring_meta.json")
    print(json.dumps(meta, indent=2))
    print(f"Shared one-time scoring setup runtime (extraction only): {extract_s:.1f}s")


if __name__ == "__main__":
    run()
