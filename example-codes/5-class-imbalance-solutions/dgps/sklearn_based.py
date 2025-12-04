"""Module for data generating procsses derived from scikit-learn functions.

Classes:
    - SKImbalancedTwoClassesDGP: binary classification with possible imbalance.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class SKImbalancedTwoClassesDGP:
    """A DGP for 2-class classification problem with possible class imbalance.

    Based on sklearn.datasets.make_classification().

    Attributes:
        n_train_samples (int): number of points in training set. Defaults to 600.
        n_test_samples (int): number of points in test set. Defaults to 200.
        weights (list[float] | np.ndarray): class weights. Defaults to [0.9, 0.1]
        num_features (int): number of classification features. Defaults to 2.
        name (str): "SK unbalanced"
    """

    def __init__(
        self,
        n_train_samples: int = 600,
        n_test_samples: int = 200,
        weights: list[float] | np.ndarray = [0.9, 0.1],
        num_features: int = 2,
    ):
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.num_features = num_features
        self.weights = weights
        self.name = "SK unbalanced"

    def sample(self, seed: int | None = None) -> list[np.ndarray]:
        """Sample from DGP with given seed.

        Args:
            seed (int | None, optional): RNG seed. Defaults to None.

        Returns:
            list[np.ndarray]: list of X_train, X_test, y_train, y_test arrays.
        """
        X, y = make_classification(
            n_samples=self.n_train_samples,
            n_features=self.num_features,
            n_redundant=0,
            weights=self.weights,
            random_state=seed,
        )
        prop_test_set = self.n_test_samples / (
            self.n_test_samples + self.n_train_samples
        )
        return train_test_split(
            X, y, test_size=prop_test_set, random_state=seed, stratify=y
        )
