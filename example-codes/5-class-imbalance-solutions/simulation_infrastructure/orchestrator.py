"""
Module for executing many Monte Carlo scenarios.

Classes
    - SimulationOrchestrator: execute list of scenarios
"""

from simulation_infrastructure.runner import SimulationRunner
from simulation_infrastructure.scenarios import SimulationScenario


class SimulationOrchestrator:
    """Simulation orchestrator that runs scenarios and stores results in a dictionary.

    Args:
        scenarios (list[SimulationScenario]): list of SimulationScenario objects
            that encode DGPs, algorithms, and associated arguments.
        n_workers (int): number of threads for parallel execution.
    """

    def __init__(
        self,
        scenarios: list[SimulationScenario],
        n_workers: int = 1,
    ):
        self.scenarios = scenarios
        self.summary_results = {}
        self.n_workers = n_workers

    def run_all(self) -> None:
        """Run all simulation scenarios"""
        for scenario in self.scenarios:
            runner = SimulationRunner(
                dgp_type=scenario.dgp,
                dgp_kwargs=scenario.dgp_kwargs,
                algorithm_types=scenario.algorithms,
                algorithm_kwargs_list=scenario.algorithm_kwargs_list,
                n_simulations=scenario.n_simulations,
                n_workers=self.n_workers,
            )
            results_df = runner.run_all()
            self.summary_results[scenario.name] = results_df
