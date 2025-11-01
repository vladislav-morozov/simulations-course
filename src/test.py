import numpy as np


def simulate_ols(
    n_sim: int = 1000,
    n_obs: int = 50,
    beta0: float = 0.0,
    beta1: float = 0.5,
    seed: int = 1,
    dgp_type: str = "static",
) -> list[float]:
    """Simulates OLS estimation for static or AR(1) DGP.
    Args:
        n_sim (int): Number of simulations to run. Defaults to 1000.
        n_obs (int): Number of observations per simulation. Defaults to 100.
        beta0 (float): True intercept value. Defaults to 0.
        beta1 (float): True slope coefficient. Defaults to 0.5
        seed (int): random number generator seed. Defaults to 1.
        dgp_type (str): Type of DGP: "static" or "ar1". Defaults to "static".

    Returns:
        list[float]: List of errors of beta1 estimates for each simulation.
    """

    # Create container to store results
    results_ols_errors = []

    # InitializeRNG
    rng = np.random.default_rng(seed)

    # Core simulation loop
    for _ in range(n_sim):
        # 1. Draw data
        if dgp_type == "static":
            x = rng.normal(size=n_obs)
            u = rng.normal(size=n_obs)
            y = beta0 + beta1 * x + u
        elif dgp_type == "ar1":
            y = np.zeros(n_obs + 1)
            u = rng.normal(size=n_obs + 1)
            y[0] = beta0 + u[0]
            for t in range(1, n_obs + 1):
                y[t] = beta0 + beta1 * y[t - 1] + u[t]
            # Use the last n_obs elements of y as and the first n_obs as x
            x = y[:-1]  # x is y lagged (n_obs elements)
            y = y[1:]  # y[1:] is the dependent variable (n_obs elements)
        else:
            raise ValueError("Invalid DGP choice")

        # 2. Run OLS estimation
        W = np.column_stack([np.ones(n_obs), x])
        beta_hat = np.linalg.inv(W.T @ W) @ W.T @ y

        # 3. Compute error
        results_ols_errors.append(beta_hat[1] - beta1)

    return results_ols_errors
