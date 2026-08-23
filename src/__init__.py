"""src — data selection below a compute-optimality threshold.

Pythia-160m (frozen scorer) -> Pythia-410m (LoRA target) on Alpaca subsets.
See cli.py for the command surface; config.py for every fixed variable.
"""
from src import config

__all__ = ["config"]
__version__ = "1.0.0"
