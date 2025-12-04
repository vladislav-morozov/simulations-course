"""
Module for multiple tests for testing joint hypotheses.

Classes:
    BonferronigMultipleTWithOLS: OLS-based multiple t-test that all non-intercept
        coefficients are zero. Uses Bonferroni correction.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests


class BonferronigMultipleTWithOLS:
    """Class for applying a multiple t-test to test a null of zero coefficients.

    Tailored to linear model
        Y_i = theta0 + theta1 X_{i1} + ... + thetap X_{ip} + U_i

    Tests the null:
        H0: theta1 = ... = thetap = 0

    Attributes:
        name (str): test name, "Bonferroni t"
        decision (np.bool): whether null is rejected
    """

    def __init__(self) -> None:
        self.name: str = "Bonferroni t"
        self.decision: np.bool

    def test(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """Carry out the multiple t-test with given data.

        Args:
            x (pd.DataFrame): covariates, including a leading constant column.
            y (pd.DataFrame): outcomes.
        """
        # Fit models
        lin_reg = OLS(y, x)
        lin_reg_fit = lin_reg.fit()

        # Perform multiple t-tests
        p_vals_t = lin_reg_fit.pvalues.iloc[1:]
        t_test_corrected_bonf = multipletests(
            p_vals_t,
            method="bonferroni", 
        )
        self.decision = t_test_corrected_bonf[0].sum() > 0
