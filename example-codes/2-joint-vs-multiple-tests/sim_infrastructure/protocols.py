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
import pandas as pd


class TestProtocol(Protocol):
    def test(self, x: np.ndarray, y: np.ndarray) -> None: ...

    @property
    def decision(self) -> float: ... 

    @property
    def name(self) -> str: ... 


class DGPProtocol(Protocol):
    def sample(
        self, n_obs: int, seed: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @property
    def covar_corr(self) -> str: ... 

    @property
    def common_coef_val(self) -> str: ... 
