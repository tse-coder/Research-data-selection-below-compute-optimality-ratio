"""LoRA fine-tune Pythia-410m on a selected subset; evaluate on the fixed
800-example held-out split.

Fixed variables: LoRA rank 8 on query_key_value (Pythia/GPT-NeoX's fused
attention projection), alpha 16, dropout 0.05, 3 epochs, LR 1e-4 cosine,
batch 8 x grad-accum 4, max_seq_len 512. Same eval split for every run.

Output: results/runs/{method}_k{k}_seed{seed}.json with eval loss, bootstrap
95% CI over per-example eval losses, selected example/token counts, selection
runtime, training runtime, and total.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src import config
from src.training.evaluation import bootstrap_ci, per_example_eval_losses


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_texts(tokenizer, texts, max_len=config.MAX_SEQ_LEN):
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    return enc["input_ids"], enc["attention_mask"]


def make_dataset(tokenizer, texts) -> Dataset:
    input_ids, attention_masks = tokenize_texts(tokenizer, texts)
    return Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_masks}
    )


def collator(tokenizer) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def run(method: str, k: int, seed: int) -> dict:
    config.ensure_dirs()
    device = get_device()
    pool = config.load_pool()
    eval_rows = config.load_eval()
    selection = config.load_selection(method, k, seed)
    indices = selection["indices"]

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_rows = [pool[i] for i in indices]
    train_texts = [
        config.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        for r in train_rows
    ]
    eval_texts = [
        config.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        for r in eval_rows
    ]

    train_ds = make_dataset(tokenizer, train_texts)

    torch.manual_seed(config.TRAIN_SEED)
    np.random.seed(config.TRAIN_SEED)

    model = AutoModelForCausalLM.from_pretrained(config.TARGET_MODEL).to(device)
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config.RUNS_DIR / f"{method}_k{k}_seed{seed}_ckpt",
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=config.TRAIN_SEED,
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

    adapter_dir = config.RUNS_DIR / f"{method}_k{k}_seed{seed}_adapter"
    model.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

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
        "fixed_config": config.fixed_config(),
    }
    path = config.RUNS_DIR / f"{method}_k{k}_seed{seed}.json"
    config.save_json(run_json, path)
    print(f"Saved {path}")
    print(
        f"[{method} k={k} seed={seed}] eval_loss={run_json['eval_loss']:.4f} "
        f"CI=[{bootstrap['ci_low']:.4f},{bootstrap['ci_high']:.4f}] "
        f"train={training_runtime_s:.1f}s selection={selection['selection_runtime_s']:.3f}s"
    )
    return run_json
