"""
Module for executing many Monte Carlo scenario.

This module contains the SimulationOrchestrator class, which executes a
supplied collection of SimulationScenario objects.

Classes:
    SimulationOrchestrator: runs simulation scenarios and stores summaries.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from sim_infrastructure.runner import SimulationRunner
from sim_infrastructure.scenarios import SimulationScenario


class SimulationOrchestratorSequential:
    """Sequential simulation orchestrator class."""

    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results = []

    def run_single_scenario(self, scenario):
        """Run a single scenario and return the result dictionary."""
        dgp = scenario.dgp(**scenario.dgp_params)
        estimator = scenario.test(**scenario.test_params)
        runner = SimulationRunner(dgp, estimator)
        runner.simulate(
            n_sim=scenario.n_simulations,
            n_obs=scenario.sample_size,
            first_seed=scenario.first_seed,
        )
        return runner.summarize_results()

    def run_all(self):
        for scenario in tqdm(self.scenarios, desc="Running simulations"):
            self.summary_results.append(self.run_single_scenario(scenario))


class SimulationOrchestratorParallel:
    """Parallelized simulation orchestrator class."""

    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results = []

    def run_single_scenario(self, scenario):
        """Run a single scenario and return the result dictionary."""
        dgp = scenario.dgp(**scenario.dgp_params)
        estimator = scenario.test(**scenario.test_params)
        runner = SimulationRunner(dgp, estimator)
        runner.simulate(
            n_sim=scenario.n_simulations,
            n_obs=scenario.sample_size,
            first_seed=scenario.first_seed,
        )
        return runner.summarize_results()

    def run_all(self, max_workers=None):
        """Run all scenarios in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenarios to the executor
            futures = {
                executor.submit(self.run_single_scenario, scenario): scenario
                for scenario in self.scenarios
            }
            # Collect results as they complete, with a progress bar
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running simulations"
            ):
                self.summary_results.append(future.result())
