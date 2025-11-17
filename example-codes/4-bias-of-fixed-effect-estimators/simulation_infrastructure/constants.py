"""
This module contains the constants used for specifying the simulation process

Constants:
- N_REPLICATIONS (int): Number of replications for each seed. the simulation.
- OUTPUT_DIR (str): Directory where the simulation results will be saved.
- SEEDS (list of int): List of seeds for random number generation to ensure reproducibility.
"""

from pathlib import Path

N_REPLICATIONS = 50
SEEDS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
OUTPUT_DIR = Path() / "results"
