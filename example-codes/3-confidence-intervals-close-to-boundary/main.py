"""
Entry point for running simulations and generating plots.

Simulation 3: evaluating confidence intervals for the variance of the normal
distribution based on the maximum likelihood estimator. Simulations consider
coverage and (scaled) length properties as true variance decreases to zero.

This script:
- Sets up simulation parameters.
- Runs the core simulation.
- Generates and saves all plots.
"""

from pathlib import Path

import numpy as np

from sim_infrastructure.core import run_simulations
from sim_infrastructure.plotting import plot_coverage_and_lengths, plot_error_kdes

# Simulation parameters: note using upper case for constants
TRUE_VARS = np.logspace(-3, 1, 50)
N_OBS_LIST = [20, 100]
N_SIMS = 50000
CI_ALPHA = 0.05
SEED = 1
OUTPUT_DIR = Path() / "results"


def main():
    # Run simulations
    results = run_simulations(TRUE_VARS, N_SIMS, N_OBS_LIST, CI_ALPHA, SEED)

    # Generate plots
    plot_coverage_and_lengths(results, OUTPUT_DIR)
    plot_error_kdes(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
