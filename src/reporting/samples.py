"""Generate 5 model outputs per completed condition into results/samples.md.

Loads the saved LoRA adapter for each run on the Pythia-410m base and answers
a fixed set of 5 eval-set instructions. Output is explicitly labeled anecdotal
inspection only, not a formal metric. Base-model responses included as a
reference row.
"""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config

N_SAMPLES = 5
MAX_NEW_TOKENS = 128
GEN_SEED = 42

CLOSING_NOTE = (
    "---\nAll of the above is explicitly anecdotal (plan Required outputs); "
    "no formal metric is derived from it."
)


def build_prompt(instruction: str, inp: str) -> str:
    parts = [config.PROMPT_INTRO, f"### Instruction:\n{instruction}\n\n"]
    if inp and inp.strip():
        parts.append(f"### Input:\n{inp}\n\n")
    parts.append("### Response:\n")
    return "".join(parts)


def generate(model, tokenizer, prompt: str, device) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=config.MAX_SEQ_LEN).to(device)
    torch.manual_seed(GEN_SEED)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main(limit: int | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    eval_rows = config.load_eval()[:N_SAMPLES]
    prompts = [
        build_prompt(r["instruction"], r.get("input", "")) for r in eval_rows
    ]

    base = AutoModelForCausalLM.from_pretrained(config.TARGET_MODEL).to(device).eval()

    lines = ["# Qualitative samples (anecdotal inspection only)\n",
             f"Generated with seed={GEN_SEED}, greedy decoding, "
             f"max {MAX_NEW_TOKENS} new tokens.\n"]

    def emit(tag: str, adapter_dir=None):
        lines.append(f"\n## {tag}\n")
        model = base
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(base, adapter_dir)
            header = f"{tag}  (eval_loss={loss_by_adapter.get(str(adapter_dir), 'n/a')})"
            lines[-1] = f"\n## {header}\n"
        for i, p in enumerate(prompts):
            resp = generate(model, tokenizer, p, device)
            lines.append(f"**Q{i+1}:** {eval_rows[i]['instruction']}"
                         + (f" | input: {eval_rows[i].get('input','')}" if eval_rows[i].get("input") else ""))
            lines.append("")
            lines.append(f"> {resp}")
            lines.append("")

    loss_by_adapter = {}
    runs = []
    for path in sorted(config.RUNS_DIR.glob("*_adapter")):
        stem = path.name[: -len("_adapter")]
        run_json = path.parent / f"{stem}.json"
        loss = "n/a"
        if run_json.exists():
            import json

            loss = f"{json.load(open(run_json))['eval_loss']:.4f}"
        loss_by_adapter[str(path)] = loss
        runs.append((stem, path))

    if limit:
        runs = runs[:limit]

    emit(f"BASE MODEL (no fine-tuning) — reference row")
    for stem, path in runs:
        emit(stem.replace("_", " "), adapter_dir=path)

    lines.append(CLOSING_NOTE)
    out = config.RESULTS_DIR / "samples.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
