"""
Protocols for DGPs and classification algorithms
"""


import numpy as np
from typing import Protocol
 
class DGPProtocol(Protocol): 
    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test)."""
        ...

class AlgorithmProtocol(Protocol):
    """Protocol for prediction algorithms.""" 
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: 
        ...
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        ... 

    @property
    def name(self) -> str: ...