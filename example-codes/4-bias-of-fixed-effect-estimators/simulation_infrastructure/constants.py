"""
This module contains the constants used for specifying the simulation process

Constants:
- N_REPLICATIONS (int): Number of replications for each seed. the simulation.
- OUTPUT_DIR (str): Directory where the simulation results will be saved.
- SEEDS (list of int): List of seeds for random number generation to ensure reproducibility.
"""

from pathlib import Path

import numpy as np

N_REPLICATIONS = 250
SEEDS = N_REPLICATIONS * np.arange(8)
OUTPUT_DIR = Path() / "results"
