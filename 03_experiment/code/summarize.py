"""Aggregate per-run results into the two required tabular outputs:

- results/runs.json — raw per-run numbers for every completed run
  (method, k, seed, eval loss, bootstrap CI, selected example/token counts,
   selection runtime, training runtime, total runtime, overhead ratio)
- results/summary.csv   — one row per (method, k, seed)

Only runs that have a saved results/runs/{method}_k{k}_seed{seed}.json are
included; the file lists which were found vs. missing.
"""
from __future__ import annotations

import csv
import json

import common


def collect_runs() -> list[dict]:
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


def main():
    common.ensure_dirs()
    runs = collect_runs()

    if not runs:
        print("No completed runs found under results/runs/. Run run_sweep.py first.")
        return

    common.save_json({"runs": runs}, common.RESULTS_DIR / "runs.json")

    csv_path = common.RESULTS_DIR / "summary.csv"
    fields = [
        "method", "k", "seed", "eval_loss", "eval_perplexity",
        "bootstrap_ci_low", "bootstrap_ci_high",
        "selected_example_count", "selected_token_count_raw",
        "selected_token_count_effective", "mean_tokens_per_example_raw",
        "selection_runtime_s", "training_runtime_s", "total_runtime_s",
        "selection_overhead_ratio", "peak_memory_gb", "device",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            row = {k: r.get(k, "") for k in fields}
            row["bootstrap_ci_low"] = r["bootstrap"]["ci_low"]
            row["bootstrap_ci_high"] = r["bootstrap"]["ci_high"]
            writer.writerow(row)

    print(f"Wrote {common.RESULTS_DIR / 'runs.json'} ({len(runs)} runs)")
    print(f"Wrote {csv_path}")
    for r in sorted(runs, key=lambda x: (x["method"], x["seed"])):
        print(f"  [{r['method']:<16} k={r['k']} seed={r['seed']:<3}] "
              f"loss={r['eval_loss']:.4f} CI=[{r['bootstrap']['ci_low']:.4f},"
              f"{r['bootstrap']['ci_high']:.4f}] "
              f"overhead={r['selection_overhead_ratio']:.4f} total={r['total_runtime_s']:.1f}s")


if __name__ == "__main__":
    main()