"""
Module for static data-generating processes (DGPs).

This module contains classes for generating static (non-time-series) data in
the model:
    Y = b0 + b1 * X + U

All classes implement the DGPProtocol.

Classes:
    StaticNormalDGP: A DGP for static linear models with normal variables.
"""

import numpy as np


class StaticNormalDGP:
    """A data-generating process (DGP) for a static linear model

    Attributes:
        beta0 (float): Intercept term. Defaults on 0.0.
        beta1 (float): Slope coefficient. Defaults to 0.5.
    """

    def __init__(self, beta0: float = 0.0, beta1: float = 0.5) -> None:
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
