"""Aggregate per-run results into results/runs.json and results/summary.csv."""
from __future__ import annotations

import csv
import json

from src import config


def collect_runs() -> list[dict]:
    runs = []
    for path in sorted(config.RUNS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "eval_loss" not in data:
            continue
        runs.append(data)
    return runs


def main() -> None:
    config.ensure_dirs()
    runs = collect_runs()

    if not runs:
        print("No completed runs found under results/runs/. Run the sweep first.")
        return

    config.save_json({"runs": runs}, config.RESULTS_DIR / "runs.json")

    csv_path = config.RESULTS_DIR / "summary.csv"
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

    print(f"Wrote {config.RESULTS_DIR / 'runs.json'} ({len(runs)} runs)")
    print(f"Wrote {csv_path}")
    for r in sorted(runs, key=lambda x: (x["method"], x["seed"])):
        print(f"  [{r['method']:<16} k={r['k']} seed={r['seed']:<3}] "
              f"loss={r['eval_loss']:.4f} CI=[{r['bootstrap']['ci_low']:.4f},"
              f"{r['bootstrap']['ci_high']:.4f}] "
              f"overhead={r['selection_overhead_ratio']:.4f} total={r['total_runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
