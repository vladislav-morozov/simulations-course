"""
This module contains functions to run Monte Carlo simulations for different
seeds and save the results to CSV files.

Functions:
    - run_simulation_for_seed(seed: int,
                            n_replications: int,
                            n_values: list[int],
                            beta_mean: float,
                            mu_sigma_params: dict[str, np.ndarray],
                            output_dir: str):
        Runs Monte Carlo for a given seed and saves the results
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from dgp.data_generation import generate_data


def run_simulation_for_seed(
    seed: int,
    n_replications: int,
    n_values: np.ndarray,
    beta_mean: float,
    mu_sigma_params: dict[str, np.ndarray | np.floating],
):
    """
    Runs Monte Carlo simulations for a specific seed and saves results to a CSV file.

    Parameters:
    - seed (int): Random seed for reproducibility.
    - n_replications (int): Number of replications per seed.
    - n_values (np.array): Cross-sectional sample sizes to simulate.
    - beta_mean (float): Average coefficient value for generating data.
    - mu_sigma_params (dict[str, np.ndarray]): Dictionary containing:
            - "mu_plus" (np.ndarray): Mean for covariates when effect is +1.
            - "mu_minus" (np.ndarray): Mean for covariates when effect is -1.
            - "sigma_plus" (np.ndarray): Covariance for X when effect is +1.
            - "sigma_minus" (np.ndarray): Covariance for X when effect is -1.
    """
    results = []

    for n_units in n_values:
        for replication in range(n_replications):
            # Generate data
            data = generate_data(
                n_units,
                beta_mean,
                mu_sigma_params,
                seed=seed + replication,
            )

            fit_no_effect = pf.feols(
                "outcome ~ covariate", data=data, drop_intercept=True
            )
            fit_effect = pf.feols("outcome ~ covariate|Unit", data=data)

            # Collect results
            results.append(
                {
                    "seed": seed,
                    "replication": replication,
                    "n_units": n_units,
                    "model": "No Fixed Effects",
                    "coef_est": fit_no_effect.coef().iloc[0],
                    "ci_lower": fit_no_effect.confint().iloc[0, 0],
                }
            )
            results.append(
                {
                    "seed": seed,
                    "replication": replication,
                    "n_units": n_units,
                    "model": "With Fixed Effects",
                    "coef_est": fit_effect.coef().iloc[0],
                    "ci_lower": fit_effect.confint().iloc[0, 0],
                }
            )

    # Return results as a dataframe
    return pd.DataFrame(results)
