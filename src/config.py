"""Shared configuration: constants, paths, prompt template, JSON helpers.

Every experimental number is defined once here and logged into run records
(the "fixed once, logged, not tuned" rule). Artifact locations default to
<repo>/artifacts/{data,selections} and <repo>/results and can be overridden
with the SRC_DATA_DIR / SRC_RESULTS_DIR environment variables.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("SRC_DATA_DIR", REPO_ROOT / "artifacts" / "data"))
RESULTS_DIR = Path(os.environ.get("SRC_RESULTS_DIR", REPO_ROOT / "results"))
RUNS_DIR = RESULTS_DIR / "runs"
SELECTIONS_DIR = DATA_DIR / "selections"

DATASET_NAME = "tatsu-lab/alpaca"
POOL_SIZE = 8000
EVAL_SIZE = 800
SPLIT_SEED = 42

TOKENIZER_NAME = "EleutherAI/pythia-160m"
PROXY_MODEL = "EleutherAI/pythia-160m"
TARGET_MODEL = "EleutherAI/pythia-410m"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query_key_value"]

NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine"
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
TRAIN_SEED = 42

MAX_SEQ_LEN = 512
TRUNCATION = True
PADDING = "batch"  # per-batch padding, left-truncated to MAX_SEQ_LEN
SCORE_BATCH_SIZE = 32

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

NEAR_DUP_GRAM = 5
NEAR_DUP_THRESHOLD = 0.8
NEAR_DUP_CAND_CAP = 2000

PRIMARY_METHODS = ["random", "length_filtered", "representation"]
STRETCH_METHODS = ["proxy_loss"]
GRADIENT_NORM_METHOD = "gradient_norm"
ALL_METHODS = PRIMARY_METHODS + STRETCH_METHODS + [GRADIENT_NORM_METHOD]
PRIMARY_K = 800
SECONDARY_K = 2000
EXTRA_SEED = 123

PROMPT_INTRO = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
)


def format_example(instruction: str, input_text: str, output: str) -> str:
    parts = [PROMPT_INTRO, f"### Instruction:\n{instruction}\n\n"]
    if input_text and input_text.strip():
        parts.append(f"### Input:\n{input_text}\n\n")
    parts.append(f"### Response:\n{output}")
    return "".join(parts)


def fixed_config() -> dict:
    return {
        "dataset": DATASET_NAME,
        "pool_size": POOL_SIZE,
        "eval_size": EVAL_SIZE,
        "split_seed": SPLIT_SEED,
        "target_model": TARGET_MODEL,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler": LR_SCHEDULER,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "train_seed": TRAIN_SEED,
        "max_seq_len": MAX_SEQ_LEN,
        "truncation": TRUNCATION,
        "padding": PADDING,
        "tokenizer": TOKENIZER_NAME,
        "proxy_model": PROXY_MODEL,
        "score_batch_size": SCORE_BATCH_SIZE,
        "eval_loss_scope": "full-sequence mean CE over non-padding tokens",
    }


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_pool() -> list[dict]:
    path = DATA_DIR / "pool.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src prepare` first")
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_eval() -> list[dict]:
    path = DATA_DIR / "eval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src prepare` first")
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_token_counts() -> list[int]:
    import numpy as np

    path = DATA_DIR / "pool_token_counts.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src prepare` first")
    return np.load(path).astype(int).tolist()


def load_selection(method: str, k: int, seed: int) -> dict:
    path = SELECTIONS_DIR / f"{method}_k{k}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `python -m src select --method {method} --k {k} --seed {seed}`"
        )
    with open(path) as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
