"""
Module for joint tests for joint hypothesie.

Classes:
    WaldWithOLS: OLS-based Wald test that all non-intercept coefficients are zero.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS


class WaldWithOLS:
    """Class for applying Wald test to test null of zero coefficients.

    Tailored to linear model
        Y_i = theta0 + theta1 X_{i1} + ... + thetap X_{ip} + U_i
        
    Tests the null:
        H0: theta1 = ... = thetap = 0

    Attributes:
        name (str): test name, "Wald"
        decision (np.bool): whether null is rejected
    """

    def __init__(self) -> None:
        self.name: str = "Wald"
        self.decision: np.bool

    def test(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """Carry out the Wald test with given data.

        Args:
            x (pd.DataFrame): covariates, including a leading constant column.
            y (pd.DataFrame): outcomes.
        """
        # Fit models
        lin_reg = OLS(y, x)
        lin_reg_fit = lin_reg.fit()

        # Perform Wald test
        num_covars = x.shape[1]
        wald_restriction_matrix = np.concatenate(
            (np.zeros((num_covars - 1, 1)), np.eye(num_covars - 1)), axis=1
        )

        wald_test = lin_reg_fit.wald_test(
            wald_restriction_matrix,
            use_f=False,
            scalar=True,
        )

        self.decision = wald_test.pvalue <= 0.05
