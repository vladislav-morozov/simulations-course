import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
from pathlib import Path
from typing import List, Dict, Tuple

def mle_variance(data: np.ndarray) -> float:
    """Estimate the variance of a normal distribution using MLE."""
    return np.var(data, ddof=0)  # MLE for variance is the sample variance with ddof=0

def asymptotic_ci_variance(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Construct an asymptotic (Wald-type) confidence interval for the variance."""
    n = len(data)
    var_hat = mle_variance(data)
    se = np.sqrt(2 * var_hat**2 / n)
    z_crit = norm.ppf(1 - alpha / 2)
    lower = max(0, var_hat - z_crit * se)  # Ensure lower bound is non-negative
    upper = var_hat + z_crit * se
    return lower, upper

def run_simulations(
    true_vars: List[float],
    n_simulations: int = 1000,
    n_obs_list: List[int] = [20, 100],
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[int, Dict[float, Dict]]:
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

def plot_coverage_and_lengths(
    results: Dict[int, Dict[float, Dict]],
    output_dir: str = "results"
) -> None:
    """Plot coverage and normalized interval lengths vs. true variance."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    true_vars = list(next(iter(results.values())).keys())

    # Coverage plot
    plt.figure(figsize=(10, 6))
    for n_obs, res in results.items():
        coverages = [res[var]['coverage'] for var in true_vars]
        plt.plot(true_vars, coverages, 'o-', label=f'n_obs={n_obs}')
    plt.axhline(0.95, color='gray', linestyle='--', label='Nominal 95%')
    plt.xscale('log')
    plt.xlabel("True Variance")
    plt.ylabel("Empirical Coverage")
    plt.title("Coverage vs. True Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/coverage_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Normalized length plot
    plt.figure(figsize=(10, 6))
    for n_obs, res in results.items():
        lengths = [np.mean(res[var]['lengths']) for var in true_vars]
        plt.plot(true_vars, lengths, 'o-', label=f'n_obs={n_obs}')
    plt.xscale('log')
    plt.xlabel("True Variance")
    plt.ylabel("Avg. Normalized CI Length")
    plt.title("Normalized CI Length vs. True Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/length_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_error_kdes(
    results: Dict[int, Dict[float, Dict]],
    output_dir: str = "results"
) -> None:
    """Plot KDE of estimation errors for each true variance and sample size."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for n_obs, res in results.items():
        for true_var, data in res.items():
            plt.figure(figsize=(10, 6))
            error_std = np.sqrt(2 * true_var**2 / n_obs)  # Asymptotic std of MLE
            x = np.linspace(min(data['errors']), max(data['errors']), 1000)
            plt.hist(data['errors'], bins=30, density=True, alpha=0.5, label='MC Errors')
            plt.plot(x, norm.pdf(x, 0, error_std), 'r-', lw=2, label='Asymptotic Normal')
            plt.title(f"Estimation Errors (n_obs={n_obs}, True Var={true_var:.4f})")
            plt.xlabel("MLE - True Variance")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/errors_n{n_obs}_var_{true_var:.4f}.png", dpi=200, bbox_inches='tight')
            plt.close()

# Main execution
if __name__ == "__main__":
    true_vars = np.logspace(-3, 1, 50)  # Logarithmic grid from 1e-3 to 10
    results = run_simulations(true_vars, n_obs_list=[20, 100])
    plot_coverage_and_lengths(results)
    plot_error_kdes(results)
