"""
Core simulation logic for evaluating confidence intervals for variance.

This module contains the core functions for:
- Maximum likelihood estimation of variance.
- Construction of asymptotic confidence intervals.
- Running Monte Carlo simulations for coverage and interval length.
"""

import numpy as np

from scipy.stats import norm

def mle_variance(data: np.ndarray) -> np.floating:
    """Estimate the variance of a normal distribution using MLE."""
    return np.var(data, ddof=0)  # MLE for variance is the sample variance with ddof=0

def asymptotic_ci_variance(data: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Construct an asymptotic (Wald-type) confidence interval for the variance."""
    n = len(data)
    var_hat = mle_variance(data)
    se = np.sqrt(2 * var_hat**2 / n)
    z_crit = norm.ppf(1 - alpha / 2)
    lower = max(0, var_hat - z_crit * se)  # Ensure lower bound is non-negative
    upper = var_hat + z_crit * se
    return lower, upper

def run_simulations(
    true_vars: np.ndarray, 
    n_simulations: int = 1000,
    n_obs_list: list[int] = [20, 100],
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[int, dict[float, dict]]:
    """
    Run Monte Carlo simulations for each true variance and sample size.

    Args:
        true_vars: List of true variance values to simulate.
        n_simulations: Number of Monte Carlo simulations per true variance.
        n_obs_list: List of sample sizes to simulate.
        alpha: Significance level for confidence intervals.
        seed: Random seed for reproducibility.

    Returns:
        Nested dictionary: {n_obs: {true_var: {'coverage': ..., 'errors': ..., 'lengths': ...}}}
    """
    np.random.seed(seed)
    results = {n_obs: {} for n_obs in n_obs_list}

    for n_obs in n_obs_list:
        for true_var in true_vars:
            errors = []
            lengths = []
            coverage = []
            for _ in range(n_simulations):
                data = np.random.normal(0, np.sqrt(true_var), n_obs)
                var_hat = mle_variance(data)
                ci_lower, ci_upper = asymptotic_ci_variance(data, alpha)
                errors.append(var_hat - true_var)
                lengths.append((ci_upper - ci_lower) / true_var)  # Normalized length
                coverage.append(ci_lower <= true_var <= ci_upper)

            results[n_obs][true_var] = {
                'coverage': np.mean(coverage),
                'errors': np.array(errors),
                'lengths': np.array(lengths)
            }

    return results