"""Efficiency-frontier plot: eval loss (y) vs. total runtime (x), one point per
(method, k, seed), colored by method. Reads results/runs/*.json.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402


def collect_runs() -> list[dict]:
    runs = []
    for path in sorted(common.RUNS_DIR.glob("*.json")):
        if path.name.endswith("_ckpt"):
            continue
        with open(path) as f:
            data = json.load(f)
        if "eval_loss" in data:
            runs.append(data)
    return runs


def main():
    common.ensure_dirs()
    runs = collect_runs()
    if not runs:
        print("No completed runs found under results/runs/.")
        return

    colors = {
        "random": "#1f77b4",
        "length_filtered": "#ff7f0e",
        "representation": "#2ca02c",
        common.GRADIENT_NORM_METHOD: "#d62728",
    }
    markers = {
        "random": "o",
        "length_filtered": "s",
        "representation": "^",
        common.GRADIENT_NORM_METHOD: "D",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in colors:
        pts = [r for r in runs if r["method"] == method]
        if not pts:
            continue
        xs = [r["total_runtime_s"] for r in pts]
        ys = [r["eval_loss"] for r in pts]
        ax.scatter(xs, ys, label=method, c=colors[method], marker=markers[method], s=90)
        for r in pts:
            ax.annotate(
                f"k={r['k']}\ns{r['seed']}",
                (r["total_runtime_s"], r["eval_loss"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    ax.set_xlabel("Total runtime (selection + training), seconds")
    ax.set_ylabel("Held-out eval loss (mean CE)")
    ax.set_title("Efficiency frontier: eval loss vs. total runtime")
    ax.legend(title="Selection method")
    ax.grid(alpha=0.3)

    out = common.RESULTS_DIR / "efficiency_frontier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out} ({len(runs)} points)")


if __name__ == "__main__":
    main()