from dgps.dynamic import DynamicNormalDGP
from estimators.ols_like import LassoWrapper
from runners import SimulationRunner

if __name__ == "__main__":
    # Initialize components
    dgp = DynamicNormalDGP(beta0=0.0, beta1=0.95)
    estimator = LassoWrapper(reg_param=0.04)

    # Run simulation
    runner = SimulationRunner(dgp, estimator)
    runner.simulate(n_sim=1000, n_obs=50, first_seed=1)

    # Print results
    print(
        f"Bias for {dgp.__class__.__name__} + {estimator.__class__.__name__}: "
    )
    runner.summarize_bias()
