"""
Module for static data-generating processes (DGPs).

This module contains classes for generating data from simple linear models.

Classes:
    BivariateLinearModel: A DGP for static linear models with normal variables,
        adjustable correlation between covariates, and two variables + constant.
"""

import numpy as np
import pandas as pd

class BivariateLinearModel:
    """A data-generating process (DGP) for a static linear model

    Attributes:
        common_coef_val (float): Common values for theta1 and theta2
        covar_corr (float): Correlation coefficient between covariates
    """

    def __init__(self, common_coef_val: float, covar_corr: float) -> None:
        """Initializes the DGP with intercept and slope.

        Args:
            beta0 (float): Intercept term. Defaults to 0.0.
            beta1 (float): Slope coefficient. Defaults to 1.0.
        """
        self.common_coef_val: float = common_coef_val
        self.covar_corr: float = covar_corr

    def sample(
        self, n_obs: int, seed: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Samples data from the static DGP.

        Args:
            n_obs (int): Number of observations to sample.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            tuple: (x, y) DataFrames, each of length n_obs.
        """
        # Initialize RNG
        rng = np.random.default_rng(seed)

        # Construct mean and covariance for coefficients
        x_mean = np.array([1, 0, 0])
        x_covar = np.array([[0, 0, 0], [0, 1, self.covar_corr], [0, self.covar_corr, 1]])

        # Construct vector of coefficients
        thetas = np.array([1, self.common_coef_val, self.common_coef_val]
                          )
        # Draw covariates and residuals, combine into output 
        covariates = rng.multivariate_normal(
            x_mean,
            x_covar,
            size=n_obs,
        )
        resids = rng.normal(0, np.sqrt(1), size=n_obs)
        y = (covariates @ thetas) + resids

        # Convert output into pandas dataframes with dynamic variable names for X
        covariates_df = pd.DataFrame(
            covariates,
            columns=[f"X{i}" for i in range(covariates.shape[1])],
        )
        y_df = pd.DataFrame(
            y,
            columns=["y"],
        )
        return covariates_df, y_df