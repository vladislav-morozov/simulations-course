"""
Module for dynamic data-generating processes (DGPs).

This module contains classes for generating dynamic (time-series) data,
such as the DynamicNormalDGP, which implements a model with single outcome and
single covariate

Classes:
    DynamicNormalDGP: A DGP for dynamic linear models with normal variables.
"""

import numpy as np


class DynamicNormalDGP:
    """A data-generating process (DGP) for a dynamic linear model: y_t = beta0 + beta1*y_{t-1} + u_t.

    Attributes:
        beta0 (float): Intercept term.
        beta1 (float): AR(1) coefficient.
    """

    def __init__(self, beta0: float = 0.0, beta1: float = 0.5):
        """Initializes the DGP with intercept and AR(1) coefficient.

        Args:
            beta0 (float): Intercept term. Defaults to 0.0.
            beta1 (float): AR(1) coefficient. Defaults to 0.5.
        """
        self.beta0: float = beta0
        self.beta1: float = beta1

    def sample(
        self, n_obs: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples data from the dynamic DGP.

        Args:
            n_obs (int): Number of observations to sample.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            tuple: (x, y) arrays, each of length n_obs.
                  x is y_{t-1} (lagged y), and y is y_t.
        """
        rng = np.random.default_rng(seed)
        y = np.zeros(n_obs + 1)  # Extra observation for lag
        u = rng.normal(size=n_obs + 1)
        y[0] = self.beta0 + u[0]  # Initial condition
        for t in range(1, n_obs + 1):
            y[t] = self.beta0 + self.beta1 * y[t - 1] + u[t]
        # Return lagged y as x and y[1:] as y
        return y[:-1], y[1:]
