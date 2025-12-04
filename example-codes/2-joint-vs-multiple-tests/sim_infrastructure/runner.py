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
        dgp (DGPProtocol): data-generating process with a sample() method and
            attributes that describe the DGP (covar_corr and common_coef_val).
        test (TestProtocol): test with a test() method and attributes that
            describe test name and test decision after seeing the data.
        test_decisions (np.ndarray): array of decisions made by the test in
            each Monte Carlo replication.
    """

    def __init__(
        self,
        dgp: DGPProtocol,
        test: TestProtocol,
    ) -> None:
        self.dgp: DGPProtocol = dgp
        self.test: TestProtocol = test
        self.test_decisions: np.ndarray = np.empty(0)

    def simulate(self, n_sim: int, n_obs: int, first_seed: int | None = None) -> None:
        """Runs simulations and stores test decisions.

        Args:
            n_sim (int): number of simulations to run.
            n_obs (int): number of sample points in each Monte Carlo dataset.
            first_seed (int | None): starting random seed for reproducibility.
                Defaults to None.
        """
        # Preallocate array to hold estimation errors
        self.test_decisions = np.empty(n_sim)

        # Run simulation
        for sim_id in range(n_sim):
            # Draw data
            x, y = self.dgp.sample(
                n_obs, seed=first_seed + sim_id if first_seed else None
            )
            # Test null of zero coefficients
            self.test.test(x, y)
            # Store error
            self.test_decisions[sim_id] = self.test.decision

    def summarize_results(self) -> dict:
        """Return a summary of results: test name, DGP characteristics, power.

        Returns:
            dict: power of current test at current DGP
        """
        return {
            "test": self.test.name,
            "rho": self.dgp.covar_corr,
            "common_coef_val": self.dgp.common_coef_val,
            "power": self.test_decisions.mean(),
        }
