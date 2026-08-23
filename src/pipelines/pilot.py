"""Pilot: end-to-end feasibility check + full-sweep runtime estimate.

Runs one real training condition (random, k=800, seed 42), measures runtimes
and memory, then estimates the full primary-sweep duration from the measured
per-run cost. Writes results/pilot_report.json.
"""
from __future__ import annotations

import time

from src import config
from src.data import prepare, scoring
from src.selection.registry import run_selection
from src.pipelines.sweep import build_plan
from src.training import train


def main() -> None:
    print("=" * 70)
    print("PILOT: random k=800 seed 42, plus full-primary-sweep runtime estimate")
    print("=" * 70)

    t_start = time.perf_counter()

    split_meta = prepare.build()
    print(f"split_meta: pool={split_meta['pool_size']} eval={split_meta['eval_size']} "
          f"eval near-dup dropped={split_meta['eval_near_dup_dropped_total']}")

    scoring.run()
    with open(config.DATA_DIR / "scoring_meta.json") as f:
        scoring_meta = __import__("json").load(f)
    shared_setup_s = scoring_meta["extract_runtime_s"]

    plan = build_plan()
    selection_records = []
    for item in plan:
        rec = run_selection(item["method"], item["k"], item["seed"])
        selection_records.append(rec)
        print(f"selection [{rec['method']} k={rec['k']} seed={rec['seed']}] "
              f"runtime={rec['selection_runtime_s']:.3f}s "
              f"tokens={rec['selected_token_count_effective']}")

    pilot_condition = {"method": "random", "k": config.PRIMARY_K, "seed": config.SPLIT_SEED}
    train_run = train.run(**pilot_condition)
    train_s = train_run["training_runtime_s"]

    runs_est = []
    for rec in selection_records:
        est = train_s * (rec["selected_token_count_effective"] / train_run["selected_token_count_effective"])
        runs_est.append({
            "method": rec["method"], "k": rec["k"], "seed": rec["seed"],
            "estimated_train_s": est,
            "selection_s": rec["selection_runtime_s"],
            "estimated_run_s": est + rec["selection_runtime_s"],
        })

    report = {
        "pilot_condition": pilot_condition,
        "measured": {
            "training_runtime_s": train_s,
            "peak_memory_gb": train_run["peak_memory_gb"],
            "selection_runtime_random_s": next(
                r["selection_runtime_s"] for r in selection_records if r["method"] == "random"
            ),
            "shared_scoring_setup_s": shared_setup_s,
            "total_pilot_wall_s": time.perf_counter() - t_start,
        },
        "sweep": {
            "n_runs": len(plan),
            "runs": runs_est,
            "estimated_total_s": sum(r["estimated_run_s"] for r in runs_est) + len(plan) * shared_setup_s,
        },
    }
    config.save_json(report, config.RESULTS_DIR / "pilot_report.json")
    print("=" * 70)
    print(f"Pilot done in {report['measured']['total_pilot_wall_s']:.1f}s; "
          f"estimated full sweep: {report['sweep']['estimated_total_s']/60:.1f} min")
    print(f"Wrote {config.RESULTS_DIR / 'pilot_report.json'}")


if __name__ == "__main__":
    main()
