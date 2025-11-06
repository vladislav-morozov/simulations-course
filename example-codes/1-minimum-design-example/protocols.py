"""
Module for defining simulation protocols.

This module contains Protocol classes that define the interfaces for
data-generating processes (DGPs) and estimators. These protocols ensure
compatibility between components in the simulation framework for analyzing
simple linear-like estimators with one covariate.

Protocols:
    DGPProtocol: Interface for data-generating processes.
    EstimatorProtocol: Interface for estimators.
"""


from typing import Protocol

import numpy as np


class EstimatorProtocol(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    @property
    def beta1_hat(self) -> float: ...


class DGPProtocol(Protocol):
    def sample(
        self, n_obs: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def beta1(self) -> float: ...
