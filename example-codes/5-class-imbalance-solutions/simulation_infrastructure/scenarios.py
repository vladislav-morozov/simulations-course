"""
Simulation scenarios for evaluating prediction algorithms.

This module defines a `SimulationScenario` dataclass and creates a list of scenarios
for evaluating algorithms under different class imbalance conditions.

Classes:
    SimulationScenario: data class for describing simulation scenarios

Variables:
    scenarios (list): list of scenarios to run.
"""

from dataclasses import dataclass
from typing import Any, Type

from algorithms.logistic import LogisticRegressionSK, LogisticRegressionSMOTE
from dgps.sklearn_based import SKImbalancedTwoClassesDGP


@dataclass(frozen=True)
class SimulationScenario:
    """A single simulation scenario: DGP, list of algorithms, and associated arguments"""

    name: str  # For readability
    dgp: Type
    dgp_kwargs: dict[str, Any]
    algorithms: list[Type]
    algorithm_kwargs_list: list[dict[str, Any]]
    n_simulations: int = 1000


# Define scenarios
scenarios = [
    # Strongly unbalanced classes
    SimulationScenario(
        name="unbalanced_classes",
        dgp=SKImbalancedTwoClassesDGP,
        dgp_kwargs={
            "n_train_samples": 600,
            "n_test_samples": 200,
            "weights": [0.9, 0.1],
        },
        algorithms=[
            LogisticRegressionSK,
            LogisticRegressionSK,
            LogisticRegressionSMOTE,
        ],
        algorithm_kwargs_list=[
            {"class_weight": None},
            {"class_weight": "balanced"},
            {},
        ],
    ),
    # Balanced classes
    SimulationScenario(
        name="balanced_classes",
        dgp=SKImbalancedTwoClassesDGP,
        dgp_kwargs={
            "n_train_samples": 600,
            "n_test_samples": 200,
            "weights": [0.5, 0.5],
        },
        algorithms=[
            LogisticRegressionSK,
            LogisticRegressionSK,
            LogisticRegressionSMOTE,
        ],
        algorithm_kwargs_list=[
            {"class_weight": None},
            {"class_weight": "balanced"},
            {},
        ],
    ),
]
