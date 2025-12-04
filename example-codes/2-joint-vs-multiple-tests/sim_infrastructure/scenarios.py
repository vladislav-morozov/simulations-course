"""
Module for defining simulation scenarios.

This module contains scenarios for comparing power functions of tests under model:
    Y = theta0  + theta1*X1 + theta2*X2 + U

The null being tested is
    H0: theta1=theta2=0

Classes:
    SimulationScenario: data class for scenarios

Variables:
    scenarios (list): list of scenarios to un.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from dgps.linear import BivariateLinearModel
from sim_infrastructure.protocols import DGPProtocol, TestProtocol
from tests.joint import WaldWithOLS
from tests.multiple import BonferronigMultipleTWithOLS


@dataclass(frozen=True)
class SimulationScenario:
    """A single simulation scenario: DGP, test, and sample size."""

    name: str  # For readability
    dgp: type[DGPProtocol]
    dgp_params: dict  # E.g. thetas go here
    test: type[TestProtocol]
    test_params: dict  #
    sample_size: int
    n_simulations: int = 500
    first_seed: int = 1


# Create DGP combinations indexed by correlation and coefficient values
common_coef_vals = np.linspace(-1.75, 1.75, 231)
covar_corr_vals = np.linspace(-0.99, 0.99, 51)
dgps = [
    (
        BivariateLinearModel,
        {"common_coef_val": common_coef_val, "covar_corr": covar_corr},
    )
    for common_coef_val, covar_corr in product(common_coef_vals, covar_corr_vals)
]

# Create list of tests
tests = [
    (WaldWithOLS, {}),
    (BonferronigMultipleTWithOLS, {}),
]

sample_sizes = [200]

# Generate all combinations
scenarios = [
    SimulationScenario(
        name="",
        dgp=dgp_class,
        dgp_params=dgp_params,
        test=test_class,
        test_params=test_params,
        sample_size=size,
    )
    for (dgp_class, dgp_params), (
        test_class,
        test_params,
    ), size in product(dgps, tests, sample_sizes)
]
