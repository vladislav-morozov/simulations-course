"""
Module for executing many Monte Carlo scenario.

This module contains the SimulationOrchestrator class, which executes a
supplied collection of SimulationScenario objects.

Classes:
    SimulationOrchestrator: runs simulation scenarios and stores summaries.
"""

from runner import SimulationRunner
from scenarios import SimulationScenario


class SimulationOrchestrator:
    """Simulation orchestrator that stores results in a dictionary

    Attributes:
        scenarios (list[SimulationScenario]): information about simulation
            scenario: DGP and estimator type, associated parameters, number of
            simulations to run, sample size, first seed.
        summary_results (dict): simulation results for each scenario, as returned
            by corresponding simulation runner
    """

    def __init__(self, scenarios: list[SimulationScenario]):
        self.scenarios = scenarios
        self.summary_results = {}

    def run_all(self):
        for scenario in self.scenarios:
            # Create DGP and estimator
            dgp = scenario.dgp(**scenario.dgp_params)
            estimator = scenario.estimator(**scenario.estimator_params)

            # Run the simulation
            runner = SimulationRunner(dgp, estimator)
            runner.simulate(
                n_sim=scenario.n_simulations,
                n_obs=scenario.sample_size,
                first_seed=scenario.first_seed,
            )
            # Save results
            self.summary_results[scenario.name] = runner.errors.mean()
