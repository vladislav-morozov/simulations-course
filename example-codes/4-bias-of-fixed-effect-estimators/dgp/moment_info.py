"""
This module defines the moment conditions determining the simulation DGP.

The moment conditions encode

1. Consistency of OLS
2. Inconsistency of FE estimators.

The module additionally includes constraints on DGP parameters,
an initial guess, and a function for processing parameters into a dictionary
expected by generate_data.

The components of this file correspond to the attributes of the GMMSolver class.

Functions:
- sim_moment_conditions(params: np.ndarray) -> np.ndarray:
    Defines the moment equations evaluated at given parameters.
- process_mu_sigma_params(params: np.ndarray) -> dict:
    Extracts mu and sigma matrices from the parameter vector.

Variables:
- constraints: constraints on the DGP parameters
- param_initial_guess: initial guess for parameters

"""

import numpy as np


def sim_moment_conditions(params: np.ndarray) -> np.ndarray:
    """Moment conditions for consistency of OLS and inconsistency of FE estimators.

    Args:
        params (np.ndarray): Parameter values.

    Returns:
        np.ndarray: Moment equations evaluated at given parameters.
    """
    sigma1p, sigma2p, rhop, mu1p, mu2p, sigma1m, sigma2m, rhom, mu1m, mu2m = params

    return np.array(
        [
            sigma2p**2 + mu2p**2 + mu2p - sigma2m**2 - mu2m**2 - mu2m,
            sigma1p**2 + mu1p**2 + mu1p - sigma1m**2 - mu1m**2 - mu1m,
            sigma1p**2
            + sigma2p**2
            - 2 * rhop * sigma1p * sigma2p
            + mu2p**2
            + mu1p**2
            - 2 * mu1p * mu2p
            - sigma1m**2
            - sigma2m**2
            + 2 * rhom * sigma1m * sigma2m
            - mu2m**2
            - mu1m**2
            + 2 * mu1m * mu2m
            - 1000,
        ]
    )


def process_mu_sigma_params(params: np.ndarray) -> dict:
    """Extracts mu and sigma matrices from the parameters vector.

    Args:
        params (np.ndarray): Estimated parameter values.

    Returns:
        dict: Processed parameters (mu and sigma matrices).
    """
    mu_plus = params[3:5]
    mu_minus = params[8:]

    sigma_plus = np.array(
        [
            [params[0] ** 2, (params[0] * params[1]) * params[2]],
            [(params[0] * params[1]) * params[2], params[1] ** 2],
        ]
    )
    sigma_minus = np.array(
        [
            [params[5] ** 2, (params[5] * params[6]) * params[7]],
            [(params[5] * params[6]) * params[7], params[6] ** 2],
        ]
    )

    return {
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
        "sigma_plus": sigma_plus,
        "sigma_minus": sigma_minus,
    }


# Constraints on data-generating process parameters
constraints = [
    {"type": "ineq", "fun": lambda vars: vars[0]},  # sigma_{1+} >= 0
    {"type": "ineq", "fun": lambda vars: vars[1]},  # sigma_{2+} >= 0
    {"type": "ineq", "fun": lambda vars: 1 - vars[2]},  # rho_+ <= 1
    {"type": "ineq", "fun": lambda vars: vars[2] + 1},  # rho_+ >= -1
    {"type": "ineq", "fun": lambda vars: vars[5]},  # sigma_{1-} >= 0
    {"type": "ineq", "fun": lambda vars: vars[6]},  # sigma_{2-} >= 0
    {"type": "ineq", "fun": lambda vars: 1 - vars[7]},  # rho_- <= 1
    {"type": "ineq", "fun": lambda vars: vars[7] + 1},  # rho_- >= -1
]

# Initial guess for parameters
param_initial_guess = np.array([12, 15, 0.3, 24, -7, 2, 8, 0.6, 5, 14])
