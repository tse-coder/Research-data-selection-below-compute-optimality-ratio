"""Efficiency-frontier plot: eval loss (y) vs total runtime (x), one point
per completed run, colored by method. Reads results/runs/*.json."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402


def collect_runs() -> list[dict]:
    runs = []
    for path in sorted(config.RUNS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "eval_loss" in data:
            runs.append(data)
    return runs


def main() -> None:
    import json

    config.ensure_dirs()
    runs = collect_runs()
    if not runs:
        print("No completed runs found under results/runs/. Run the sweep first.")
        return

    methods = sorted({r["method"] for r in runs})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in methods:
        pts = [(r["total_runtime_s"], r["eval_loss"], f"{m}\nseed {r['seed']}") for r in runs if r["method"] == m]
        xs, ys, labels = zip(*pts)
        ax.scatter(xs, ys, label=m, s=70)
        for x, y, lab in pts:
            ax.annotate(lab.replace("\n", " s"), (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Total runtime: selection + training (s)")
    ax.set_ylabel("Eval loss (lower is better)")
    ax.set_title("Efficiency Frontier — Eval Loss vs Total Runtime")
    ax.legend(title="Method")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = config.RESULTS_DIR / "efficiency_frontier.png"
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
