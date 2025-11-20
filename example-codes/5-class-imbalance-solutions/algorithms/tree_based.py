"""Algorithms based on trees.

Classes:
    - RandomForestAlgorithm - vanilla sklearn random forest classifier.

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestAlgorithm:
    """Random forest classifier.

    Attributes:
        model (sklearn.ensemble.RandomForestClassifier): fitted random forest model.
    """

    def __init__(
        self,
        random_state: int | None = None,
    ) -> None:
        self.name = "RandomForest"
        self.model = RandomForestClassifier(random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the random forest model to the training data.

        Args:
            X (np.ndarray): training features.
            y (np.ndarray): training labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the test data.

        Args:
            X (np.ndarray): test features.

        Returns:
            np.ndarray: predicted labels.
        """
        return self.model.predict(X)
