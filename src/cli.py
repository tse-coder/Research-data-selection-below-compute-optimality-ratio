"""Single CLI entry point for the whole experiment.

Usage:
    python -m src <command> [options]

Commands:
    prepare     Build the fixed pool/eval splits from Alpaca
    score       One-time frozen-scorer extraction over the pool
    select      Run a data-selection method
    train       LoRA fine-tune on a selected subset and evaluate
    sweep       Full primary sweep (5 runs, resumable)
    pilot       End-to-end feasibility check + runtime estimate
    summarize   Aggregate run JSONs into runs.json / summary.csv
    plot        Efficiency-frontier figure
    samples     Qualitative generations per condition (anecdotal)
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="src",
        description="Data selection below a compute-optimality threshold.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build pool/eval splits")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("score", help="Frozen-scorer extraction (shared setup)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--with-proxy-loss", action="store_true")

    p = sub.add_parser("select", help="Run one selection method")
    p.add_argument("--method", required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity-check", action="store_true",
                   help="(gradient_norm only) check the first --sanity-n examples")
    p.add_argument("--sanity-n", type=int, default=20)
    p.add_argument("--diagnostics", action="store_true",
                   help="(gradient_norm only) save per-example diagnostics")

    p = sub.add_parser("train", help="LoRA fine-tune + evaluate")
    p.add_argument("--method", required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("sweep", help="Full primary sweep")
    p.add_argument("--force", action="store_true")

    sub.add_parser("pilot", help="Feasibility check + runtime estimate")
    sub.add_parser("summarize", help="Aggregate results")
    sub.add_parser("plot", help="Efficiency-frontier figure")

    p = sub.add_parser("samples", help="Qualitative generations")
    p.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    dispatch(args)


def dispatch(args) -> None:
    from src import config

    if args.command == "prepare":
        from src.data import prepare
        prepare.main(force=args.force)

    elif args.command == "score":
        from src.data import scoring
        scoring.run(force=args.force, with_proxy_loss=args.with_proxy_loss)

    elif args.command == "select":
        from src.selection import registry

        if args.sanity_check:
            if args.method != config.GRADIENT_NORM_METHOD:
                raise SystemExit("--sanity-check is only valid with --method gradient_norm")
            registry.gradient_norm_sanity_check(n=args.sanity_n)
            return
        registry.run_selection(args.method, args.k or config.PRIMARY_K, args.seed,
                               save_diagnostics=args.diagnostics)

    elif args.command == "train":
        from src.training import train
        train.run(args.method, args.k or config.PRIMARY_K, args.seed)

    elif args.command == "sweep":
        from src.pipelines import sweep
        sweep.main(force=args.force)

    elif args.command == "pilot":
        from src.pipelines import pilot
        pilot.main()

    elif args.command == "summarize":
        from src.reporting import summarize
        summarize.main()

    elif args.command == "plot":
        from src.reporting import frontier
        frontier.main()

    elif args.command == "samples":
        from src.reporting import samples
        samples.main(limit=args.limit)


if __name__ == "__main__":
    main()
