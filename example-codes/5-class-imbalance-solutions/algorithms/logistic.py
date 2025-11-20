"""
Algorithms based on logistic regression, with and without class proportion
corrections.

Classes in this module:
    - LogisticRegressionAlgorithm: vanilla scikit-learn regression
    - LogisticRegressionSMOTE: logistic regression with SMOTE
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression


class LogisticRegressionAlgorithm:
    """Logistic regression classifier with optional class weighting.

    Attributes:
        name (str): name for reporting purposes.
        class_weight (str | None): optional class weights for imbalanced datasets.
        model (sklearn.linear_model.LogisticRegression): logistic regression model.
    """

    def __init__(
        self,
        class_weight: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.class_weight = class_weight
        self.model = LogisticRegression(
            class_weight=class_weight, random_state=random_state
        )
        self.name = f"LogisticRegression (class_weight={self.class_weight})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model to the training data.

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


class LogisticRegressionSMOTE:
    """Logistic regression classifier with SMOTE oversampling.

    Attributes:
        name (str): name for reporting purposes.
        model (ImbPipeline): Pipeline combining SMOTE oversampling and logistic regression.
    """

    def __init__(
        self,
        random_state: int | None = None,
    ) -> None:
        """Initialize the logistic regression with SMOTE algorithm."""
        self.name = "Logistic Regression with SMOTE"
        self.model = ImbPipeline(
            [
                ("smote", SMOTE(random_state=random_state)),
                ("logistic", LogisticRegression(random_state=random_state)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model with SMOTE to the training data.

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
            np.ndarray: tredicted labels.
        """
        return self.model.predict(X)
