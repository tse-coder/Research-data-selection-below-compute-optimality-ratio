# Data Selection Below a Compute-Optimality Threshold

A small-scale empirical study of LLM data-efficient fine-tuning: does
intelligent training-data selection pay for itself when the scorer-to-learner
model-size ratio sits *below* the ~5–10x threshold where prior work says
selection becomes worthwhile?

**Setup:** Pythia-160m (frozen scorer) → Pythia-410m (LoRA target), pool of
8,000 Alpaca examples, train on k=800 (10%), evaluated on 800 held-out
questions. Four selection methods compared against random at identical
training budgets.

## Repository structure

```
├── Makefile                  # runner: `make help` lists every target
├── pyproject.toml            # packaging metadata
├── requirements.txt
└── src/                      # the delivered package
    ├── __main__.py           # `python -m src <command>` entry point
    ├── cli.py                # command surface (prepare/score/select/train/…)
    ├── config.py             # every fixed variable + artifact paths (env-overridable)
    ├── data/
    │   ├── prepare.py        # build fixed pool/eval splits + near-dup filter
    │   └── scoring.py        # one-time frozen-scorer extraction
    ├── selection/
    │   ├── base.py           # seeding, tie-break ranks, Spearman
    │   ├── random_selector.py
    │   ├── length_filtered_selector.py
    │   ├── representation_selector.py
    │   ├── proxy_loss_selector.py
    │   ├── gradient_norm_selector.py   # GraNd arm + sanity check + diagnostics
    │   └── registry.py       # dispatch, timing, selection record schema
    ├── training/
    │   ├── evaluation.py     # per-example eval losses + bootstrap CI
    │   └── train.py          # LoRA fine-tune + evaluate per condition
    ├── reporting/
    │   ├── summarize.py      # run JSONs -> runs.json / summary.csv
    │   ├── frontier.py       # efficiency-frontier figure
    │   └── samples.py        # qualitative generations
    └── pipelines/
        ├── pilot.py          # feasibility check + runtime estimate
        └── sweep.py          # full primary sweep (resumable)
```

Experiment provenance (design document, decisions log, archived raw outputs)
lives in `03_experiment/`; the write-up in `04_writeup/`. Both are working
records, not part of the deliverable package.

## Inputs / Outputs

Artifact locations default to `artifacts/data` and `results/`, overridable via
the `SRC_DATA_DIR` / `SRC_RESULTS_DIR` environment variables.

**Inputs** (`artifacts/data/`):

| File | Description |
|---|---|
| `pool.jsonl` | 8,000 training-pool examples (Alpaca, seed-42 shuffle) |
| `eval.jsonl` | 800 held-out exam questions, near-duplicate-filtered |
| `pool_token_counts.npy` | untruncated token length per pool example |
| `pool_reprs.npy` | cached Pythia-160m final-layer vectors (8,000 × 1024) |
| `selections/*.json` | picked index lists per method/k/seed |
| `split_meta.json`, `scoring_meta.json` | provenance for split and scoring |

**Outputs** (`results/`):

| File | Description |
|---|---|
| `runs/*.json` | one report card per run: eval loss, bootstrap 95% CI, token counts, timings |
| `runs/*_adapter/` | trained LoRA adapters |
| `runs.json` / `summary.csv` | all runs aggregated (raw / spreadsheet) |
| `efficiency_frontier.png` | eval loss vs total runtime, per method |
| `samples.md` | qualitative model generations (anecdotal only) |
| `pilot_report.json` | pre-sweep feasibility measurements |

## Running

```bash
pip install -r requirements.txt

make pilot        # end-to-end feasibility check + runtime estimate
make sweep        # full primary sweep (5 runs, resumable)
make all          # sweep + summarize + plot + samples
```

Single condition / finer control:

```bash
make sanity                                    # gradient-norm check (~20 examples)
make select METHOD=representation SEED=42
make train METHOD=random K=800 SEED=42
make summarize && make plot && make samples
```

Every target is a thin wrapper over the CLI — the equivalent direct form is
`python -m src <command>`, e.g.
`python -m src train --method random --k 800 --seed 42`.

## Key finding

Random selection matched or beat every intelligent method. Gradient-norm
selection was significantly worse than random while consuming ~50x more
selection compute (~81% of its total budget). Below the compute-optimality
threshold, selection overhead is not recovered — a clean, statistically
supported negative result extending Yin & Rush (2024).

## AI usage disclosure

This project was developed with assistance from AI coding
assistants (OpenCode and ChatGPT), used for experiment design discussion,
code implementation, debugging, literature synthesis, and drafting support.
All experimental decisions, results interpretation, and final content were
reviewed and approved by the author, who takes full responsibility for the
work.
