"""One-time shared setup cost: load Pythia-160m frozen once, extract
final-hidden-layer, attention-mask-aware mean-pooled, L2-normalized
representations for every pool example, and save them to data/.

Also optionally computes Pythia-160m's mean per-token loss per pool example
(--with-proxy-loss) for the stretch proxy-loss selection method.

The extraction runtime measured here is the shared one-time setup cost and is
NOT charged per-method (see plan Metrics section).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import common


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_texts(pool: list[dict]) -> list[str]:
    return [
        common.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        for r in pool
    ]


@torch.no_grad()
def extract_reprs(
    model, tokenizer, texts, device, batch_size=common.SCORE_BATCH_SIZE, max_len=common.MAX_SEQ_LEN
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
    model, tokenizer, texts, device, batch_size=common.SCORE_BATCH_SIZE, max_len=common.MAX_SEQ_LEN
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract frozen Pythia-160m pool representations (one-time shared setup)."
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    parser.add_argument(
        "--with-proxy-loss",
        action="store_true",
        help="Also compute mean per-token proxy loss per pool example (for stretch proxy-loss selection)",
    )
    args = parser.parse_args()

    common.ensure_dirs()
    repr_path = common.DATA_DIR / "pool_reprs.npy"
    loss_path = common.DATA_DIR / "pool_proxy_losses.npy"
    if not args.force and repr_path.exists() and (not args.with_proxy_loss or loss_path.exists()):
        print("Scoring output already present; use --force to recompute.")
        return

    pool = common.load_pool()
    texts = load_texts(pool)
    device = get_device()
    print(f"Device: {device}")

    t_model = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        common.PROXY_MODEL, output_hidden_states=True
    ).to(device)
    model.eval()
    model_load_s = time.perf_counter() - t_model

    t_extract = time.perf_counter()
    reprs = extract_reprs(model, tokenizer, texts, device)
    extract_s = time.perf_counter() - t_extract

    np.save(repr_path, reprs.astype(np.float32))
    print(f"Saved {repr_path} shape={reprs.shape}")

    if args.with_proxy_loss:
        t_loss = time.perf_counter()
        losses = compute_proxy_losses(model, tokenizer, texts, device)
        loss_s = time.perf_counter() - t_loss
        np.save(loss_path, losses.astype(np.float32))
        print(f"Saved {loss_path} shape={losses.shape} proxy-loss compute {loss_s:.1f}s")

    meta = {
        "model": common.PROXY_MODEL,
        "pool_size": len(pool),
        "batch_size": common.SCORE_BATCH_SIZE,
        "max_seq_len": common.MAX_SEQ_LEN,
        "tokenizer": common.TOKENIZER_NAME,
        "device": str(device),
        "pooling": "final hidden layer, attention-mask-aware mean, L2-normalized",
        "model_load_s": model_load_s,
        "extract_runtime_s": extract_s,
        "with_proxy_loss": args.with_proxy_loss,
    }
    common.save_json(meta, common.DATA_DIR / "scoring_meta.json")
    print(json.dumps(meta, indent=2))
    print(f"Shared one-time scoring setup runtime (extraction only): {extract_s:.1f}s")


if __name__ == "__main__":
    main()
