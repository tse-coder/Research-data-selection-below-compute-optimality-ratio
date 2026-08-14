"""Generate 5-10 model outputs per completed condition into results/samples.md.

Loads the saved LoRA adapter for each run (results/runs/{method}_k{k}_seed{seed}_adapter)
on the Pythia-410m base, then generates a response for a fixed set of 5 eval-set
instructions. Output is explicitly labeled "anecdotal inspection only, not a
formal metric" per the plan. Base model responses are included as a reference row.

The prompt for each sample is the eval-set instruction + input (if any) with an
empty response, so the model continues with its own answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

N_SAMPLES = 5
MAX_NEW_TOKENS = 128
GEN_SEED = 42

CLOSING_NOTE = (
    "---\nAll of the above is explicitly anecdotal (plan §Required outputs); "
    "no formal metric is derived from it."
)


def load_run_files() -> list[dict]:
    runs = []
    for path in sorted(common.RUNS_DIR.glob("*.json")):
        if path.name.endswith("_ckpt"):
            continue
        with open(path) as f:
            data = json.load(f)
        if "eval_loss" not in data:
            continue
        runs.append(data)
    return runs


def build_prompt(r: dict) -> str:
    parts = [common.PROMPT_INTRO, f"### Instruction:\n{r['instruction']}\n\n"]
    if r.get("input", "").strip():
        parts.append(f"### Input:\n{r['input']}\n\n")
    parts.append("### Response:\n")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate anecdotal samples for completed runs.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--only-method",
        action="append",
        default=None,
        help="Only generate sections for this method (repeatable); "
             "defaults to all completed runs",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new sections to an existing samples.md instead of "
             "regenerating the whole file (keeps existing content untouched)",
    )
    args = parser.parse_args()

    runs = load_run_files()
    if not runs:
        print("No completed runs found under results/runs/. Run run_sweep.py first.")
        return
    if args.only_method:
        runs = [r for r in runs if r["method"] in args.only_method]
        if not runs:
            print(f"No runs found for method(s) {args.only_method}.")
            return

    out_path = common.RESULTS_DIR / "samples.md"
    existing_text = out_path.read_text() if args.append and out_path.exists() else None

    eval_rows = common.load_eval()
    prompts = [build_prompt(r) for r in eval_rows[:N_SAMPLES]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(common.TARGET_MODEL).to(device)
    torch.manual_seed(GEN_SEED)

    def generate(model, text: str) -> str:
        enc = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    lines = []
    if existing_text is not None:
        body = existing_text.rstrip()
        if body.endswith(CLOSING_NOTE):
            body = body[: -len(CLOSING_NOTE)].rstrip()
        lines.append(body)
        lines.append("")
    else:
        with torch.no_grad():
            base_outputs = [generate(base, p) for p in prompts]
        base.eval()
        lines = [
            "# Anecdotal inspection only, not a formal metric",
            "",
            f"{len(prompts)} eval-set instructions; each row shows the base model's and one",
            "per-condition fine-tuned model's continuation. Shuffle before close reading if",
            f"time allows. Generated with seed {GEN_SEED}, do_sample, top_p=0.9, temperature=0.7.",
            "",
        ]

    for run in runs:
        name = f"{run['method']} k={run['k']} seed={run['seed']}"
        adapter_dir = common.RUNS_DIR / f"{run['method']}_k{run['k']}_seed{run['seed']}_adapter"
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"SKIP {name}: no saved adapter at {adapter_dir} "
                  f"(needs re-run with train.py that saves adapters)")
            continue
        lines.append("")
        lines.append(f"## {name}  (eval_loss={run['eval_loss']:.4f})")
        lines.append("")
        model = PeftModel.from_pretrained(base, adapter_dir).to(device)
        model.eval()
        with torch.no_grad():
            outputs = [generate(model, p) for p in prompts]
        if "base_outputs" not in locals():
            base_outputs = [generate(base, p) for p in prompts]
            base.eval()
        for i, (p, o, bo) in enumerate(zip(prompts, outputs, base_outputs)):
            inst = p.split("### Instruction:\n")[1].split("\n\n")[0]
            lines.append(f"### Sample {i + 1} — instruction: {inst!r}")
            lines.append("")
            lines.append(f"- base:      {bo}")
            lines.append(f"- fine-tuned: {o}")
            lines.append("")
        del model
        torch.cuda.empty_cache()

    lines.append(CLOSING_NOTE)

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}" + (" (appended)" if existing_text is not None else ""))


if __name__ == "__main__":
    main()