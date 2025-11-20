"""
Protocols for DGPs and classification algorithms
"""

from typing import Protocol

import numpy as np


class DGPProtocol(Protocol):
    """Protocol for data generating processes"""

    def sample(
        self, seed: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test)."""
        ...

    @property
    def n_train_samples(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def weights(self) -> list[float] | np.ndarray: ...


class AlgorithmProtocol(Protocol):
    """Protocol for prediction algorithms."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def name(self) -> str: ...
