"""Data-selection methods. One module per method; `run_selection` dispatches.

Methods:
  random          - uniform sample of k from the pool
  length_filtered - drop bottom/top 10% by token count, sample k from the middle
  representation  - greedy farthest-point (k-center) on cached L2-normalized vectors
  proxy_loss      - stratified sampling over cached proxy-loss quartiles (stretch)
  gradient_norm   - GraNd-style output-embedding gradient-norm top-k (backward pass)
"""
from __future__ import annotations

from src.selection.registry import (
    ALL_SELECTORS,
    gradient_norm_sanity_check,
    run_selection,
)

__all__ = ["ALL_SELECTORS", "run_selection", "gradient_norm_sanity_check"]
