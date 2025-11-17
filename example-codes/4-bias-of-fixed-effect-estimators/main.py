"""
Entry point for running simulation on bias of FE estimators under heterogeneity.

Causal model:
    Y_{it}^x = alpha_i + beta_i * x + U_{it}


Goal of simulation: construct an example of distribution under which:
    1. The OLS estimator in regression of Y_{it} on X_{it} is unbiased for
        average coefficient value E[beta_i]
    2. The one-way random intercept/fixed effects estimator has the wrong sign
       relative to E[beta_i] with high probability

Usage:
    python main.py

Output:
    Kernel density plots of distributions of the two estimators
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from dgp.constants import (
    BETA_MEAN,
    N_VALUES,
)

# from utils.combine_results import combine_results
from dgp.moment_info import (
    constraints,
    param_initial_guess,
    process_mu_sigma_params,
    sim_moment_conditions,
)
from gmm.solver import GMMSolver
from simulation_infrastructure.constants import (
    N_REPLICATIONS,
    OUTPUT_DIR,
    SEEDS,
)
from simulation_infrastructure.plotting import plot_kdes
from simulation_infrastructure.runner_functions import run_simulation_for_seed

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    """Main function to run simulations in parallel and combine results."""

    # Select parameters of DGP by enforcing desired moment conditions
    solver_dgp_params = GMMSolver(
        sim_moment_conditions,
        param_initial_guess,
        constraints,
        process_func=process_mu_sigma_params,
    )
    solver_dgp_params.minimize()
    mu_sigma_params = solver_dgp_params.process_solution()
 

    # Run simulations in parallel
    all_results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_simulation_for_seed,
                seed,
                N_REPLICATIONS,
                N_VALUES,
                BETA_MEAN,
                mu_sigma_params,
            )
            for seed in SEEDS
        }
        # Collect results as they complete, with a progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Running simulations"
        ):
            result = future.result()
            all_results.append(result)

    # Combine results and export plots
    sim_results = pd.concat(all_results)
    plot_kdes(sim_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
