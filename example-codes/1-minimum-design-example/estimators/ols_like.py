"""
Module for OLS-like estimators.

This module contains classes for estimating simple model
    Y = b0 + b1 * X + U
with OLS-like methods.

All estimators implement the EstimatorProtocol interface.

Classes:
    SimpleOLS: ordinary least squares estimator.
    SimpleRidge: ridge regression estimator.
    LassoWrapper: wrapper for scikit-learn's Lasso estimator.
"""

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso


class SimpleOLS:
    """A simple OLS estimator for the linear model y = beta0 + beta1*x + u.

    Attributes:
        beta0_hat (float): Estimated intercept. NaN until fit is called.
        beta1_hat (float): Estimated slope. NaN until fit is called.
    """

    def __init__(self) -> None: 
        self.beta0_hat: float = np.nan
        self.beta1_hat: float = np.nan

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit OLS to the provided data.

        Args:
            x (np.ndarray): Independent variable (1D array).
            y (np.ndarray): Dependent variable (1D array).
        """

        # Add constant to x
        X = np.column_stack([np.ones(len(x)), x])
        # OLS estimation
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        self.beta0_hat, self.beta1_hat = beta_hat[0], beta_hat[1]


class SimpleRidge:
    """A simple ridge estimator for the linear model y = beta0 + beta1*x + u.

    Attributes:
        beta0_hat (float): Estimated intercept. NaN until fit is called.
        beta1_hat (float): Estimated slope. NaN until fit is called.
        reg_param (float): Strength of regularization. Defaults to 0.01.
    """

    def __init__(self, reg_param: float = 0.01) -> None: 
        self.beta0_hat: float = np.nan
        self.beta1_hat: float = np.nan
        self.reg_param = reg_param

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the ridge estimator to the provided data.

        Args:
            x (np.ndarray): Independent variable (1D array).
            y (np.ndarray): Dependent variable (1D array).
        """

        # Add constant to x
        X = np.column_stack([np.ones(len(x)), x])
        # OLS estimation
        beta_hat = np.linalg.inv(X.T @ X + self.reg_param * np.eye(2)) @ X.T @ y
        self.beta0_hat, self.beta1_hat = beta_hat[0], beta_hat[1]


class LassoWrapper:
    """A wrapper for scikit-learn's Lasso to match the EstimatorProtocol.
    Model y = beta0 + beta1*x+u.

    Attributes:
        model (sklearn.linear_model.Lasso): a scikit-learn Lasso instance.
        beta0_hat (float): Estimated intercept. Initialized as np.nan.
        beta1_hat (float): Estimated slope. Initialized as np.nan.
        reg_param (float): Regularization strength (alpha). Defaults to 1.0.
    """

    def __init__(self, reg_param: float = 1.0) -> None:
        self.model = SklearnLasso(alpha=reg_param)
        self.beta0_hat: float = np.nan
        self.beta1_hat: float = np.nan
        self.reg_param = reg_param

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the Lasso model to the provided data.

        Args:
            x (np.ndarray): Independent variable (1D array).
            y (np.ndarray): Dependent variable (1D array).
        """
        x = x.reshape(-1, 1)  # sklearn expects 2d array inputs
        self.model.fit(x, y)
        self.beta0_hat = float(self.model.intercept_)
        self.beta1_hat = float(self.model.coef_[0])
