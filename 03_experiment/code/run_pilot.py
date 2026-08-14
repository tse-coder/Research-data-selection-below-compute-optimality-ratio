"""Mandatory pilot: run ONE condition (random, k=800, seed 42) end to end and
print an estimated total runtime for the full primary sweep.

Full primary sweep = 3 methods x k=800 (seed 42) plus seed-123 repeats for
random and representation = 5 training runs.

This script measures:
  - shared one-time scoring setup runtime (score_pool.py)
  - marginal selection runtime for each of the 5 selections
  - one actual training run (random, k=800, seed 42) and its peak memory
Then it extrapolates to the 5-run sweep and prints the estimate.

It deliberately does NOT run the sweep. Stop, report the estimate to the
researcher, and wait for confirmation before running run_sweep.py.
"""
from __future__ import annotations

import json
import time

import numpy as np

import common
import prepare_data
import score_pool
import selection
import train


def build_plan() -> list[dict]:
    plan = []
    for method in common.PRIMARY_METHODS:
        plan.append({"method": method, "k": common.PRIMARY_K, "seed": common.SPLIT_SEED})
    for method in ("random", "representation"):
        plan.append({"method": method, "k": common.PRIMARY_K, "seed": common.EXTRA_SEED})
    return plan


def main():
    print("=" * 70)
    print("PILOT: random k=800 seed 42, plus full-primary-sweep runtime estimate")
    print("=" * 70)

    t_start = time.perf_counter()

    split_meta = prepare_data.build()
    print(f"split_meta: pool={split_meta['pool_size']} eval={split_meta['eval_size']} "
          f"eval near-dup dropped={split_meta['eval_near_dup_dropped_total']}")

    t_score = time.perf_counter()
    score_pool.main()
    scoring_runtime_s = time.perf_counter() - t_score
    with open(common.DATA_DIR / "scoring_meta.json") as f:
        scoring_meta = json.load(f)
    shared_setup_s = scoring_meta["extract_runtime_s"]

    plan = build_plan()
    selection_records = []
    for item in plan:
        rec = selection.run_selection(item["method"], item["k"], item["seed"])
        selection_records.append(rec)
        print(f"selection [{rec['method']} k={rec['k']} seed={rec['seed']}] "
              f"runtime={rec['selection_runtime_s']:.3f}s tokens={rec['selected_token_count_effective']}")

    run_json = train.run("random", common.PRIMARY_K, common.SPLIT_SEED)
    training_runtime_s = run_json["training_runtime_s"]
    peak_mem = run_json["peak_memory_gb"]

    ref_tokens = None
    for rec in selection_records:
        if rec["method"] == "random" and rec["seed"] == common.SPLIT_SEED:
            ref_tokens = rec["selected_token_count_effective"]
    if ref_tokens is None:
        raise RuntimeError("random seed-42 selection record missing")

    per_run_estimates = []
    total_estimate_s = 0.0
    for item, rec in zip(plan, selection_records):
        train_s = training_runtime_s * (rec["selected_token_count_effective"] / ref_tokens)
        sel_s = rec["selection_runtime_s"]
        run_s = train_s + sel_s
        total_estimate_s += run_s
        per_run_estimates.append(
            {
                "method": item["method"],
                "k": item["k"],
                "seed": item["seed"],
                "estimated_train_s": train_s,
                "selection_s": sel_s,
                "estimated_run_s": run_s,
            }
        )

    report = {
        "pilot_condition": {"method": "random", "k": common.PRIMARY_K, "seed": common.SPLIT_SEED},
        "measured": {
            "training_runtime_s": training_runtime_s,
            "peak_memory_gb": peak_mem,
            "selection_runtime_random_s": selection_records[0]["selection_runtime_s"],
            "shared_scoring_setup_s": shared_setup_s,
            "total_pilot_wall_s": time.perf_counter() - t_start,
        },
        "sweep": {
            "n_runs": len(plan),
            "runs": per_run_estimates,
            "estimated_total_s": total_estimate_s,
            "estimated_total_min": total_estimate_s / 60.0,
            "plus_shared_scoring_s": shared_setup_s,
        },
    }
    common.save_json(report, common.RESULTS_DIR / "pilot_report.json")

    print("=" * 70)
    print("PILOT MEASUREMENTS")
    print(f"  training runtime (random k=800 seed 42): {training_runtime_s:.1f}s")
    print(f"  peak GPU memory: {peak_mem:.2f} GB" if peak_mem else "  peak memory: CPU (n/a)")
    print(f"  shared one-time scoring setup: {shared_setup_s:.1f}s (separate session, not per-method)")
    print(f"  marginal selection runtimes (per method):")
    for rec in selection_records:
        print(f"    {rec['method']:<16} k={rec['k']} seed={rec['seed']:<3} {rec['selection_runtime_s']:.3f}s")
    print("=" * 70)
    print("ESTIMATED FULL PRIMARY SWEEP (5 runs: 3 methods x k=800 + 2 x seed-123)")
    for e in per_run_estimates:
        print(f"  {e['method']:<16} k={e['k']} seed={e['seed']:<3} "
              f"~{e['estimated_run_s']/60:.1f} min (train ~{e['estimated_train_s']/60:.1f} + sel {e['selection_s']:.3f}s)")
    print(f"  TOTAL: ~{total_estimate_s/60:.1f} min ({total_estimate_s/3600:.2f} h)")
    print(f"  + shared scoring (already/once): ~{shared_setup_s/60:.1f} min")
    print("=" * 70)
    print("STOPPING HERE. Report this estimate to the researcher and wait for")
    print("confirmation before running the full sweep (run_sweep.py).")


if __name__ == "__main__":
    main()
