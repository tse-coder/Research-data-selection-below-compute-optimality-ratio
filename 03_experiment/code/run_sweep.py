"""Full primary sweep: 3 methods x k=800 (seed 42) plus seed-123 repeats for
random and representation = 5 training runs.

Run this only AFTER the mandatory pilot (run_pilot.py) confirmed feasibility
and the researcher approved. Runs whose output already exists are skipped
unless --force is given, so re-running after a mid-sweep disconnect resumes
without repeating finished runs.

Each run calls train.run() which writes results/runs/{method}_k{k}_seed{seed}.json.
This script just drives the sweep and prints a completion summary.
"""
from __future__ import annotations

import argparse

import common
import train


def build_plan() -> list[dict]:
    plan = []
    for method in common.PRIMARY_METHODS:
        plan.append({"method": method, "k": common.PRIMARY_K, "seed": common.SPLIT_SEED})
    for method in ("random", "representation"):
        plan.append({"method": method, "k": common.PRIMARY_K, "seed": common.EXTRA_SEED})
    return plan


def main():
    parser = argparse.ArgumentParser(description="Run the full primary sweep (5 runs).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conditions even if results/runs output already exists",
    )
    args = parser.parse_args()

    common.ensure_dirs()
    plan = build_plan()
    print(f"Sweep: {len(plan)} runs (3 methods x k={common.PRIMARY_K} seed 42 + "
          f"seed {common.EXTRA_SEED} for random and representation)")
    print("=" * 70)

    done, skipped = [], []
    for item in plan:
        out = common.RUNS_DIR / f"{item['method']}_k{item['k']}_seed{item['seed']}.json"
        if out.exists() and not args.force:
            print(f"SKIP  {item['method']:<16} k={item['k']} seed={item['seed']} "
                  f"(output exists: {out.name})")
            skipped.append(item)
            continue
        run_json = train.run(item["method"], item["k"], item["seed"])
        done.append(run_json)

    print("=" * 70)
    print(f"DONE: {len(done)} runs completed, {len(skipped)} skipped (already present).")
    for r in done:
        print(f"  [{r['method']} k={r['k']} seed={r['seed']}] "
              f"eval_loss={r['eval_loss']:.4f} total={r['total_runtime_s']:.1f}s")
    print("Next: run summarize.py to build runs.json + summary.csv, then "
          "plot_frontier.py and samples.py for the remaining required outputs.")


if __name__ == "__main__":
    main()