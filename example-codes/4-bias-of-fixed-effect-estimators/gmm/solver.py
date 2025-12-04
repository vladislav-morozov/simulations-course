"""
Implements a simple generic Generalized Method of Moments (GMM) class.

Classes:
    - GMMSolver: Estimates parameters by minimizing weighted squared distance
        of moment conditions to zero.
"""

from typing import Callable

import numpy as np
from scipy.optimize import minimize


class GMMSolver:
    """
    A Generalized Method of Moments (GMM) solver that estimates parameters by
    minimizing the weighted squared distance of moment conditions from zero.

    Attributes:
        moment_conditions (Callable[[np.ndarray], np.ndarray]): function
            returning moment conditions given parameter values.
        constraints (list[dict[str, str | Callable]] | None): list of
            constraints for parameter optimization. Defaults to None.
        initial_guess (np.ndarray): initial parameter values for the
            optimization.
        weighting_matrix (np.ndarray): weighting matrix for the GMM objective
            function. Defaults to the identity matrix.
        process_func (Callable[[np.ndarray], dict[str, np.array| np.float]] | None):
            function that processes the optimized parameters into a meaningful format.
            Defaults to NOne.
        estimated_params (np.ndarray | None):
            The estimated parameters after optimization.
    """

    def __init__(
        self,
        moment_conditions: Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
        constraints: list[dict[str, str | Callable]] | None = None,
        weighting_matrix: np.ndarray | None = None,
        process_func: Callable[[np.ndarray], dict[str, np.ndarray | np.floating]]
        | None = None,
    ) -> None:
        self.moment_conditions = moment_conditions
        self.initial_guess = np.array(initial_guess)
        self.constraints = constraints if constraints else []
        self.weighting_matrix = (
            np.eye(len(moment_conditions(initial_guess)))
            if weighting_matrix is None
            else weighting_matrix
        )
        self.process_func = process_func
        self.estimated_params: np.ndarray = np.empty(self.initial_guess.shape)

    def _gmm_objective(self, params: np.ndarray) -> float:
        """Computes the GMM objective function:
                m(θ)ᵀ W m(θ)
        where m(θ) are the moment conditions.

        Args:
            params (np.ndarray): Current parameter estimates.

        Returns:
            float: The GMM loss function value.
        """
        moments = self.moment_conditions(params)
        return float(moments.T @ self.weighting_matrix @ moments)

    def minimize(self) -> None:
        """Runs the GMM estimation by minimizing the GMM objective function."""
        result = minimize(
            self._gmm_objective, self.initial_guess, constraints=self.constraints
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        self.estimated_params = result.x

    def process_solution(self) -> dict[str, np.ndarray | np.floating]:
        """Processes the estimated parameters into a meaningful format.

        Returns:
            dict[str, Any]: Processed output, depends on user-supplied processing function.
        """
        if self.process_func is None:
            return {"parameters": self.estimated_params}
        return self.process_func(self.estimated_params)
