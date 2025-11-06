"""
Module for static data-generating processes (DGPs).

This module contains classes for generating static (non-time-series) data,
such as the StaticNormalDGP, which implements a model with single outcome and
single covariate

Classes:
    StaticNormalDGP: A DGP for static linear models with normal variables.
"""

import numpy as np


class StaticNormalDGP:
    """A data-generating process (DGP) for a static linear model

    Attributes:
        beta0 (float): Intercept term.
        beta1 (float): Slope coefficient.
    """

    def __init__(self, beta0: float = 0.0, beta1: float = 0.5) -> None:
        """Initializes the DGP with intercept and slope.

        Args:
            beta0 (float): Intercept term. Defaults to 0.0.
            beta1 (float): Slope coefficient. Defaults to 1.0.
        """
        self.beta0: float = beta0
        self.beta1: float = beta1

    def sample(
        self, n_obs: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples data from the static DGP.

        Args:
            n_obs (int): Number of observations to sample.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            tuple: (x, y) arrays, each of length n_obs.
        """
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n_obs)
        u = rng.normal(size=n_obs)
        y = self.beta0 + self.beta1 * x + u
        return x, y
