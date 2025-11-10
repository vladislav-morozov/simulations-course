"""
Module for executing a given Monte Carlo scenario.

This module contains the SimulationRunner class, which executes a scenario
by combining a DGP and a test. It handles data generation, estimation, testing,
and result collection.

Classes:
    SimulationRunner: Runs simulations and summarizes results.
"""

import numpy as np
from sim_infrastructure.protocols import DGPProtocol, TestProtocol


class SimulationRunner:
    """Runs Monte Carlo simulations for a given DGP and estimator.

    Attributes:
        dgp: data-generating process with a sample() method and beta1 attribute.
        estimator: estimator with a fit() method and beta1_hat attribute.
        errors: array of estimation errors (beta1_hat - beta1) for each simulation.
    """

    def __init__(
        self,
        dgp: DGPProtocol,
        test: TestProtocol,
    ) -> None:
        """Initializes the simulation runner.

        Args:
            dgp: An instance of a DGP class (must implement `sample`).
            estimator: An instance of an estimator class (must implement `fit`).
        """
        self.dgp: DGPProtocol = dgp
        self.test: TestProtocol = test
        self.test_decisions: np.ndarray = np.empty(0)

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
            self.test.test(x, y)
            # Store error
            self.test_decisions[sim_id] = self.test.decision

    def summarize_results(self) -> dict:
        """Return a summary of results

        Returns:
            dict: power of current test at current DGP
        """
        return {
            "test": self.test.name,
            "rho": self.dgp.covar_corr,
            "common_coef_val": self.dgp.common_coef_val,
            "power": self.test_decisions.mean(),
        }
