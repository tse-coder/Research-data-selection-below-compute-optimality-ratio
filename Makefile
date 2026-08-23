# Runner for the src experiment.
# The real interface is `python -m src <command>`; these targets are
# thin wrappers so the full pipeline is runnable with plain `make <target>`.

PYTHON ?= python3
METHOD ?= random
K      ?= 800
SEED   ?= 42

.PHONY: help prepare score select sanity train sweep pilot summarize plot samples all

help:
	@echo "Targets:"
	@echo "  make prepare                       Build pool/eval splits from Alpaca"
	@echo "  make score                         Frozen-scorer extraction (shared setup)"
	@echo "  make select METHOD=random          Run one selection method"
	@echo "  make sanity                        Gradient-norm sanity check (~20 examples)"
	@echo "  make train METHOD=random SEED=42   LoRA fine-tune + evaluate one condition"
	@echo "  make sweep                         Full primary sweep (5 runs, resumable)"
	@echo "  make pilot                         Feasibility check + runtime estimate"
	@echo "  make summarize                     Aggregate runs -> runs.json / summary.csv"
	@echo "  make plot                          Efficiency-frontier figure"
	@echo "  make samples                       Qualitative generations (anecdotal)"
	@echo "  make all                           sweep + summarize + plot + samples"
	@echo ""
	@echo "Variables: PYTHON METHOD K SEED"

prepare:
	$(PYTHON) -m src prepare

score:
	$(PYTHON) -m src score

select:
	$(PYTHON) -m src select --method $(METHOD) --k $(K) --seed $(SEED)

sanity:
	$(PYTHON) -m src select --method gradient_norm --sanity-check --sanity-n 20

diagnostics:
	$(PYTHON) -m src select --method gradient_norm --k $(K) --seed $(SEED) --diagnostics

train:
	$(PYTHON) -m src train --method $(METHOD) --k $(K) --seed $(SEED)

sweep:
	$(PYTHON) -m src sweep

pilot:
	$(PYTHON) -m src pilot

summarize:
	$(PYTHON) -m src summarize

plot:
	$(PYTHON) -m src plot

samples:
	$(PYTHON) -m src samples

all: sweep summarize plot samples
