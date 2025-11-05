from scenarios import SimulationScenario
from runner import SimulationRunner

class SimulationOrchestrator:
    """Simulation orchestrator that stores results in a dictionary
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