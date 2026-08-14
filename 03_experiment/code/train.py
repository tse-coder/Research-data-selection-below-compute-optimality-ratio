"""LoRA fine-tune Pythia-410m on a given selected subset and evaluate on the
fixed 800-example held-out split.

Fixed variables (do not change across conditions): Pythia-410m + LoRA rank 8 on
query_key_value (Pythia/GPT-NeoX's fused attention projection; the plan's
q_proj/v_proj naming assumed a Llama-style module layout), alpha 16, dropout
0.05; 3 epochs; LR 1e-4 cosine; batch size and
max sequence length fixed once and logged; same eval split for every run.

Output: results/runs/{method}_k{k}_seed{seed}.json with eval loss (full-sequence
mean CE over non-padding tokens), bootstrap 95% CI over per-example eval losses,
selected example/token counts, selection runtime, training runtime, and total.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import common


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_texts(tokenizer, texts, max_len=common.MAX_SEQ_LEN):
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    return enc["input_ids"], enc["attention_mask"]


def make_dataset(tokenizer, texts):
    input_ids, attention_masks = tokenize_texts(tokenizer, texts)
    return Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_masks}
    )


def collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


@torch.no_grad()
def per_example_eval_losses(
    model, tokenizer, texts, device, batch_size=common.EVAL_BATCH_SIZE, max_len=common.MAX_SEQ_LEN
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


def bootstrap_ci(losses: np.ndarray, n=common.BOOTSTRAP_N, seed=common.BOOTSTRAP_SEED) -> dict:
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


def run(method: str, k: int, seed: int) -> dict:
    common.ensure_dirs()
    device = get_device()
    pool = common.load_pool()
    eval_rows = common.load_eval()
    selection = common.load_selection(method, k, seed)
    indices = selection["indices"]

    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_rows = [pool[i] for i in indices]
    train_texts = [
        common.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        for r in train_rows
    ]
    eval_texts = [
        common.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        for r in eval_rows
    ]

    train_ds = make_dataset(tokenizer, train_texts)

    torch.manual_seed(common.TRAIN_SEED)
    np.random.seed(common.TRAIN_SEED)

    model = AutoModelForCausalLM.from_pretrained(common.TARGET_MODEL).to(device)
    lora_config = LoraConfig(
        r=common.LORA_RANK,
        lora_alpha=common.LORA_ALPHA,
        lora_dropout=common.LORA_DROPOUT,
        target_modules=common.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=common.RUNS_DIR / f"{method}_k{k}_seed{seed}_ckpt",
        num_train_epochs=common.NUM_EPOCHS,
        learning_rate=common.LEARNING_RATE,
        lr_scheduler_type=common.LR_SCHEDULER,
        per_device_train_batch_size=common.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=common.GRAD_ACCUM_STEPS,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=common.TRAIN_SEED,
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator(tokenizer),
    )

    t0 = time.perf_counter()
    trainer.train()
    training_runtime_s = time.perf_counter() - t0

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = float(torch.cuda.max_memory_allocated() / 1e9)

    eval_losses = per_example_eval_losses(model, tokenizer, eval_texts, device)
    bootstrap = bootstrap_ci(eval_losses)

    run_json = {
        "method": method,
        "k": int(k),
        "seed": int(seed),
        "eval_loss": float(eval_losses.mean()),
        "eval_perplexity": float(np.exp(eval_losses.mean())),
        "bootstrap": bootstrap,
        "selected_example_count": selection["selected_example_count"],
        "selected_token_count_raw": selection["selected_token_count_raw"],
        "selected_token_count_effective": selection["selected_token_count_effective"],
        "mean_tokens_per_example_raw": selection["mean_tokens_per_example_raw"],
        "selection_runtime_s": selection["selection_runtime_s"],
        "training_runtime_s": training_runtime_s,
        "total_runtime_s": selection["selection_runtime_s"] + training_runtime_s,
        "selection_overhead_ratio": (
            selection["selection_runtime_s"]
            / (selection["selection_runtime_s"] + training_runtime_s)
            if training_runtime_s > 0
            else None
        ),
        "peak_memory_gb": peak_mem_gb,
        "device": str(device),
        "fixed_config": common.fixed_config(),
    }
    path = common.RUNS_DIR / f"{method}_k{k}_seed{seed}.json"
    common.save_json(run_json, path)
    print(f"Saved {path}")
    print(
        f"[{method} k={k} seed={seed}] eval_loss={run_json['eval_loss']:.4f} "
        f"CI=[{bootstrap['ci_low']:.4f},{bootstrap['ci_high']:.4f}] "
        f"train={training_runtime_s:.1f}s selection={selection['selection_runtime_s']:.3f}s"
    )
    return run_json


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune on a selected subset and evaluate.")
    parser.add_argument("--method", required=True, choices=common.PRIMARY_METHODS + common.STRETCH_METHODS)
    parser.add_argument("--k", type=int, default=common.PRIMARY_K)
    parser.add_argument("--seed", type=int, default=common.SPLIT_SEED)
    args = parser.parse_args()
    run(args.method, args.k, args.seed)


if __name__ == "__main__":
    main()
