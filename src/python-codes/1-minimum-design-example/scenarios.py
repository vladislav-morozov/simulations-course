"""
Module for defining simulation scenarios.

This module contains scenarios for simulations on the simple model:
    y = b0 + b1 * x + u
The scenarios evaluate several OLS-like estimators under static and dynamic DGPs
with two different sample sizes.

Classes:
    SimpleOLS: ordinary least squares estimator.

Variables:
    scenarios (list): list of scenarios to un.
"""

from dataclasses import dataclass
from itertools import product

from dgps.dynamic import DynamicNormalDGP
from dgps.static import StaticNormalDGP
from estimators.ols_like import LassoWrapper, SimpleOLS, SimpleRidge
from protocols import DGPProtocol, EstimatorProtocol


@dataclass(frozen=True)
class SimulationScenario:
    """A single simulation scenario: DGP, estimator, and sample size."""

    name: str  # For readability
    dgp: type[DGPProtocol]
    dgp_params: dict  # E.g. betas go here
    estimator: type[EstimatorProtocol]
    estimator_params: dict  # E.g. reg_params go here
    sample_size: int
    n_simulations: int = 1000
    first_seed: int = 1


# Define lists of components
dgps = [
    (StaticNormalDGP, {"beta0": 0.0, "beta1": 1.0}, "static"),
    (DynamicNormalDGP, {"beta0": 0.0, "beta1": 0.1}, "low_pers"),
    (DynamicNormalDGP, {"beta0": 0.0, "beta1": 0.5}, "mid_pers"),
    (DynamicNormalDGP, {"beta0": 0.0, "beta1": 0.95}, "high_pers"),
]
estimators = [
    (SimpleOLS, {}),
    (LassoWrapper, {"reg_param": 0.1}),
    (SimpleRidge, {"reg_param": 0.1}),
]
sample_sizes = [50, 200]

# Generate all combinations
scenarios = [
    SimulationScenario(
        name=f"{dgp_class.__name__.lower()}_{dgp_descr}_{estimator_class.__name__.lower()}_n{size}",
        dgp=dgp_class,
        dgp_params=dgp_params,
        estimator=estimator_class,
        estimator_params=estimator_params,
        sample_size=size,
    )
    for (dgp_class, dgp_params, dgp_descr), (
        estimator_class,
        estimator_params,
    ), size in product(dgps, estimators, sample_sizes)
]
