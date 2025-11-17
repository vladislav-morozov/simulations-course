"""
Functions for data generation for FE model.

Functions:
    - generate_data(
            num_units: int,
            beta_mean: float,
            params: dict[str, np.ndarray],
            seed: int = None,
        ) -> pd.DataFrame:
        Generates (Y, X) panel data follow DGP in the lecture.
"""

import numpy as np
import pandas as pd


def generate_data(
    num_units: int,
    beta_mean: float,
    params: dict[str, np.ndarray | np.floating],
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generates an `num_units x 2` panel dataset with two types of slopes:
    `beta_mean + 1` and `beta_mean - 1`.

    Args:
        num_units (int): Total number of units.
        beta_mean (float): Average coefficient for the covariates.
        params (Dict[str, np.ndarray]): Dictionary containing:
            - "mu_plus" (np.ndarray): Mean for covariates when effect is +1.
            - "mu_minus" (np.ndarray): Mean for covariates when effect is -1.
            - "sigma_plus" (np.ndarray): Covariance for X when effect is +1.
            - "sigma_minus" (np.ndarray): Covariance for X when effect is -1.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated dataset.
    """

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Generate individual effects
    ind_effects = rng.choice([-1, 1], size=num_units)

    # Helper function to generate data for a given effect type
    def generate_for_effect(effect, mu_x, sigma_x, sigma_u, beta):
        # Select units with the specified effect
        num_units_effect = np.sum(ind_effects == effect)

        # Generate covariates and shocks
        covariates = rng.multivariate_normal(mu_x, sigma_x, size=num_units_effect)
        shocks = rng.normal(loc=0, scale=sigma_u, size=(num_units_effect, 2))

        # Generate outcomes
        outcomes = effect + beta * covariates + shocks

        # Create a DataFrame
        data = pd.DataFrame(
            {
                "outcome": pd.DataFrame(outcomes).stack(),
                "covariate": pd.DataFrame(covariates).stack(),
                "effects": pd.DataFrame(ind_effects).stack()
            }
        )
        data.index = data.index.rename(["Unit", "Period"])
        return data.reset_index()

    # Generate data for +1 and -1 effects
    data_plus = generate_for_effect(
        1,
        params["mu_plus"],
        params["sigma_plus"],
        1,
        beta_mean + 1,
    )
    data_minus = generate_for_effect(
        1,
        params["mu_minus"],
        params["sigma_minus"],
        1,
        beta_mean - 1,
    )

    # Adjust unit indices for negative effect data
    data_minus["Unit"] += len(data_plus["Unit"].unique())

    # Combine datasets and reset index
    return pd.concat([data_plus, data_minus], axis=0).reset_index(drop=True)
