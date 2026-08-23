"""Full primary sweep: 3 methods x k=800 (seed 42) plus seed-123 repeats for
random and representation = 5 training runs.

Runs whose output already exists are skipped unless --force, so re-running
after a mid-sweep disconnect resumes without repeating finished runs.
"""
from __future__ import annotations

import argparse

from src import config
from src.training import train


def build_plan() -> list[dict]:
    plan = []
    for method in config.PRIMARY_METHODS:
        plan.append({"method": method, "k": config.PRIMARY_K, "seed": config.SPLIT_SEED})
    for method in ("random", "representation"):
        plan.append({"method": method, "k": config.PRIMARY_K, "seed": config.EXTRA_SEED})
    return plan


def main(force: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Run the full primary sweep (5 runs).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run conditions even if results/runs output already exists")
    args = parser.parse_args()
    force = force or args.force

    config.ensure_dirs()
    plan = build_plan()
    print(f"Sweep: {len(plan)} runs (3 methods x k={config.PRIMARY_K} seed 42 + "
          f"seed-123 repeats for random and representation)")

    for i, item in enumerate(plan, 1):
        out = config.RUNS_DIR / (
            f"{item['method']}_k{item['k']}_seed{item['seed']}.json"
        )
        if out.exists() and not force:
            print(f"[{i}/{len(plan)}] SKIP {out.name} (already exists)")
            continue
        print(f"[{i}/{len(plan)}] RUN {item}")
        train.run(**item)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
