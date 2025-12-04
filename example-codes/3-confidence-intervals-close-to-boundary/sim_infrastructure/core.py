"""
Core simulation logic for evaluating confidence intervals for variance.

This module contains the following core functions:
    - mle_variance(): maximum likelihood estimator for the variance of the 
        normal distribution.
    - asymptotic_ci_variance(): construct an asymptotic confidence interval for
        the variance of the normal distribution based on the limit distribution
        of the maximum likelihood estimator for variance.
    - run_simulations(): run Monte Carlo simulations for coverage and interval
        length of confidence intervals for the variance of the normal
        distribution based on the maximum likelihood estimator.
"""

import numpy as np

from scipy.stats import norm


def mle_variance(data: np.ndarray) -> np.floating:
    """Estimate the variance of a normal distribution using maximum likelihood.

    Args:
        data (np.ndarray): 1D array with samples.

    Returns:
        np.floating: maximum likelihood estimator for variance of the normal
            distribution.
    """
    # MLE for variance is the sample variance with ddof=0
    return np.var(data, ddof=0)


def asymptotic_ci_variance(
    data: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Construct an asymptotic (Wald-type) confidence interval for the variance.

    Lower bound of the confidence interval truncated to zero if necessary.

    Args:
        data (np.ndarray): 1D array with samples.
        alpha (float, optional): 1-alpha is the desired nominal coverage.
            Defaults to 0.05 (fir 95% nominal intervals.)

    Returns:
        tuple[float, float]: lower and upper limits of the confidence interval.
    """
    n_obs = len(data)

    # Run maximum likelihood
    variance_est = mle_variance(data)
    variance_std_error = np.sqrt(2 * variance_est**2 / n_obs)

    # Construct intervals, ensuring non-negative lower bound
    z_crit = norm.ppf(1 - alpha / 2)
    lower = max(0, variance_est - z_crit * variance_std_error)
    upper = variance_est + z_crit * variance_std_error
    return lower, upper


def run_simulations(
    true_vars: np.ndarray,
    n_simulations: int = 1000,
    n_obs_list: list[int] = [20, 100],
    alpha: float = 0.05,
    seed: int = 1,
) -> dict[int, dict[float, dict]]:
    """
    Simulate confindence intervals properties for given variances and sample sizes.

    Runs a separate simulation round for each combination of (true_variance,
    n_obs), sampling n_obs points from a N(0, true_variance) distribution.

    For each scenario the simulation computes:
        - Coverage;
        - Array of all estimation errors of the maximum likelihood estimator.
        - Array of lenghts of all confidence intervals.

    Args:
        true_vars (np.ndarray): true variance values characterizing the DGPs.
        n_simulations (int): number of Monte Carlo simulations per each
            scenario. Defaults to 1000.
        n_obs_list (list[int]): sample sizes characterizing the DGPs. Defaults to
            to [20, 100]
        alpha (float, optional): 1-alpha is the desired nominal coverage.
            Defaults to 0.05 (fir 95% nominal intervals.)
        seed (int): Random seed for reproducibility. Defaults to 1.

    Returns:
        Nested dictionary: {n_obs: {true_var: {'coverage': ..., 'errors': ...,
            'lengths': ...}}}. Coverages are reported as a single float, averaged
            across Monte Carlo draws. Estimation errors and confidence interval
            lengths are arrays.
    """
    rng = np.random.default_rng(seed)
    results = {n_obs: {} for n_obs in n_obs_list}

    # Loop over sample sizes
    for n_obs in n_obs_list:
        # Loop over true variance values
        for true_var in true_vars:
            errors = []
            lengths = []
            coverage = []
            for _ in range(n_simulations):
                # Draw data and run estimation
                data = rng.normal(0, np.sqrt(true_var), n_obs)
                var_hat = mle_variance(data)

                # Record estimation error in variance
                errors.append(var_hat - true_var)

                # Construct CI and check if variance belongs to it
                ci_lower, ci_upper = asymptotic_ci_variance(data, alpha)
                lengths.append((ci_upper - ci_lower) / true_var)
                coverage.append(ci_lower <= true_var <= ci_upper)

            results[n_obs][true_var] = {
                "coverage": np.mean(coverage),
                "errors": np.array(errors),
                "lengths": np.array(lengths),
            }

    return results
