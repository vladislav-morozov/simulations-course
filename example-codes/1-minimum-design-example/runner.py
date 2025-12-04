"""
Module for executing a given Monte Carlo scenario.

This module contains the SimulationRunner class, which executes a scenario
by combining a DGP and an estimator. It handles data generation, estimation,
and result collection.

Classes:
    SimulationRunner: Runs simulations and summarizes results.
"""

import numpy as np

from protocols import DGPProtocol, EstimatorProtocol


class SimulationRunner:
    """Runs Monte Carlo simulations for a given DGP and estimator.

    Attributes:
        dgp (DGPProtocol): data-generating process with a sample() method and
            beta1 attribute.
        estimator (EstimatorProtocol): estimator with a fit() method and
            beta1_hat attribute.
        errors np.ndarray): array of estimation errors (beta1_hat - beta1) for
            each Monte Carlo draw.
    """

    def __init__(
        self,
        dgp: DGPProtocol,
        estimator: EstimatorProtocol,
    ) -> None: 
        self.dgp: DGPProtocol = dgp
        self.estimator: EstimatorProtocol = estimator
        self.errors: np.ndarray = np.empty(0)

    def simulate(self, n_sim: int, n_obs: int, first_seed: int | None = None) -> None:
        """Runs simulations and stores estimation errors.

        Args:
            n_sim (int): number of simulations to run.
            n_obs (int): Number of observations per simulation.
            first_seed (int | None): Starting random seed for reproducibility.
                Defaults to None.
        """
        # Preallocate array to hold estimation errors
        self.errors = np.empty(n_sim)

        # Run simulation
        for sim_id in range(n_sim):
            # Draw data
            x, y = self.dgp.sample(
                n_obs, seed=first_seed + sim_id if first_seed else None
            )
            # Fit model
            self.estimator.fit(x, y)
            # Store error
            self.errors[sim_id] = self.estimator.beta1_hat - self.dgp.beta1

    def summarize_bias(self) -> None:
        """Prints the average estimation error (bias) for beta1."""
        print(f"Average estimation error (bias): {self.errors.mean():.4f}")
