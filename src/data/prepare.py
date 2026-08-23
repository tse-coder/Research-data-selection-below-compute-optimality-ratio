"""Build the fixed 8,000-example pool and 800-example held-out eval split from
Alpaca (seed 42), with a mandatory near-duplicate check between pool and eval.

Near-duplicate rule: an eval candidate is dropped if its normalized
instruction+input text exactly equals a pool example's, or its char-5-gram
Jaccard overlap with any pool example is >= 0.8. Eval candidates are drawn
from the remainder of the seed-42 shuffle until 800 pass; drop counts are
logged in data/split_meta.json.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from src import config


def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def shingles(text: str, k: int = config.NEAR_DUP_GRAM) -> set:
    compact = text.replace(" ", "")
    if len(compact) < k:
        return {compact} if compact else set()
    return {compact[i : i + k] for i in range(len(compact) - k + 1)}


def _exact_norm(row: dict) -> str:
    return normalize_text(f"{row['instruction']} {row.get('input', '')}")


def build_near_dup_index(pool: list[dict]) -> tuple[set, list[set], dict]:
    exact = set()
    shingle_sets = []
    inv = {}
    for r in pool:
        norm = _exact_norm(r)
        exact.add(norm)
        ss = shingles(norm)
        shingle_sets.append(ss)
        for sh in ss:
            inv.setdefault(sh, []).append(len(shingle_sets) - 1)
    return exact, shingle_sets, inv


def is_near_dup(row: dict, exact: set, pool_shingles: list[set], inv: dict) -> bool:
    norm = _exact_norm(row)
    if norm in exact:
        return True
    ss = shingles(norm)
    if not ss:
        return False
    cand = {}
    for sh in ss:
        for i in inv.get(sh, []):
            cand[i] = cand.get(i, 0) + 1
    cand_ids = [i for i, _ in sorted(cand.items(), key=lambda x: -x[1])[: config.NEAR_DUP_CAND_CAP]]
    for i in cand_ids:
        union = len(ss | pool_shingles[i])
        if union and len(ss & pool_shingles[i]) / union >= config.NEAR_DUP_THRESHOLD:
            return True
    return False


def compute_token_counts(pool: list[dict]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    counts = []
    for r in pool:
        text = config.format_example(r["instruction"], r.get("input", ""), r.get("output", ""))
        counts.append(len(tokenizer(text)["input_ids"]))
    return counts


def build(force: bool = False) -> dict:
    config.ensure_dirs()
    meta_path = config.DATA_DIR / "split_meta.json"
    if not force and (config.DATA_DIR / "pool.jsonl").exists() and meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)

    t0 = time.perf_counter()
    ds = load_dataset(config.DATASET_NAME, split="train")
    ds = ds.shuffle(seed=config.SPLIT_SEED)
    rows = [dict(r) for r in ds]

    pool = rows[: config.POOL_SIZE]
    rest = rows[config.POOL_SIZE :]

    exact, pool_shingles, inv = build_near_dup_index(pool)

    eval_rows = []
    dropped_exact = 0
    dropped_ngram = 0
    scanned = 0
    for r in rest:
        scanned += 1
        if len(eval_rows) >= config.EVAL_SIZE:
            break
        if is_near_dup(r, exact, pool_shingles, inv):
            if _exact_norm(r) in exact:
                dropped_exact += 1
            else:
                dropped_ngram += 1
            continue
        eval_rows.append(r)

    if len(eval_rows) < config.EVAL_SIZE:
        raise RuntimeError(
            f"Only {len(eval_rows)} clean eval candidates found in {scanned} scanned "
            f"rows (need {config.EVAL_SIZE}); pool may be too large for this dataset."
        )

    token_counts = compute_token_counts(pool)

    with open(config.DATA_DIR / "pool.jsonl", "w") as f:
        for r, tc in zip(pool, token_counts):
            f.write(json.dumps({**r, "token_count_raw": tc}) + "\n")
    with open(config.DATA_DIR / "eval.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    np.save(config.DATA_DIR / "pool_token_counts.npy", np.asarray(token_counts, dtype=np.int64))

    meta = {
        "dataset": config.DATASET_NAME,
        "split_seed": config.SPLIT_SEED,
        "pool_size": len(pool),
        "eval_size": len(eval_rows),
        "eval_near_dup_dropped_exact": dropped_exact,
        "eval_near_dup_dropped_ngram": dropped_ngram,
        "eval_near_dup_dropped_total": dropped_exact + dropped_ngram,
        "eval_candidates_scanned": scanned,
        "near_dup_method": "normalized instruction+input equality OR char-5-gram Jaccard >= 0.8",
        "near_dup_threshold": config.NEAR_DUP_THRESHOLD,
        "token_count_def": "Pythia tokenizer length of formatted instruction+input+response, untruncated",
        "build_runtime_s": time.perf_counter() - t0,
    }
    config.save_json(meta, meta_path)
    print(json.dumps(meta, indent=2))
    return meta


def main(force: bool = False) -> None:
    build(force=force)


if __name__ == "__main__":
    main()
