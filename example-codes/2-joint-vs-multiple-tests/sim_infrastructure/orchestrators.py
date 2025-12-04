"""
Module for executing many Monte Carlo scenario.

Classes:
    SimulationOrchestratorParallel: runs simulations in parallel
    SimulationOrchestratorSequential: runs simulations sequentially
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from sim_infrastructure.runner import SimulationRunner
from sim_infrastructure.scenarios import SimulationScenario


class SimulationOrchestratorParallel:
    """Parallelized simulation orchestrator class.

    Attributes:
        scenarios (list[SimulationScenario]): information about simulation
            scenario: DGP and estimator type, associated parameters, number of
            simulations to run, sample size, first seed.
        summary_results (list): simulation results for each scenario, as
            returned by corresponding simulation runner.
    """

    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results = []

    def _run_single_scenario(self, scenario: SimulationScenario):
        """Run a single scenario and return the results."""

        dgp = scenario.dgp(**scenario.dgp_params)
        estimator = scenario.test(**scenario.test_params)

        runner = SimulationRunner(dgp, estimator)
        runner.simulate(
            n_sim=scenario.n_simulations,
            n_obs=scenario.sample_size,
            first_seed=scenario.first_seed,
        )

        return runner.summarize_results()

    def run_all(self, max_workers: int | None = None):
        """Run all scenarios with thread-based parallelism.
        Args:
            max_workers (int | None, optional): number of threads to use in 
                execution. Defaults to None.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenarios to the executor
            futures = {
                executor.submit(self._run_single_scenario, scenario): scenario
                for scenario in self.scenarios
            }

            # Collect results as they complete, with a progress bar
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running simulations"
            ):
                self.summary_results.append(future.result())


class SimulationOrchestratorSequential:
    """Sequential simulation orchestrator class.

    Attributes:
        scenarios (list[SimulationScenario]): information about simulation
            scenario: DGP and estimator type, associated parameters, number of
            simulations to run, sample size, first seed.
        summary_results (list): simulation results for each scenario, as
            returned by corresponding simulation runner.
    """

    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results = []

    def _run_single_scenario(self, scenario):
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
        """Run all simulation scenarios.

        The results are stored in the summary_results attribute.
        """
        for scenario in tqdm(self.scenarios, desc="Running simulations"):
            self.summary_results.append(self._run_single_scenario(scenario))
