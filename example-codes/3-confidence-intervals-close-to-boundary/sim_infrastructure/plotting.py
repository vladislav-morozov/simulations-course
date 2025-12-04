"""
Plotting functions for simulation results.

This module contains the following functions:
    - plot_coverage_and_lengths(): plot confidence interval coverage and
        normalized interval length as a function of true variance.
    - plot_error_kdes(): plot density of estimation errors for each
        value of true variance and sample size.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

# Set constants for plotting here
BG_COLOR = "whitesmoke"


def plot_coverage_and_lengths(
    results: dict[int, dict[float, dict]], output_dir: Path
) -> None:
    """Plots coverage and normalized interval lengths as a function of the
       true variance.

    The same figure is used for all sample sizes in the results.

    The expected results dictionary is compatible with the format returned by
    the sim_infrastructure.core.run_simulations() function.

    Args:
        results (dict[int, dict[float, dict]]): dictionary indexed as
            {n_obs: {true_var: {"coverage":..., "length":...}}}.
        output_dir (Path): where to store the resultin gfigures.
    """
    # Make sure output folder exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract the true variancs
    true_vars = list(next(iter(results.values())).keys())

    # Coverage plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    for n_obs, res in results.items():
        coverages = [res[var]["coverage"] for var in true_vars]
        ax.plot(true_vars, coverages, "o-", label=f"{n_obs} observations")
    ax.axhline(0.95, color="gray", linestyle="--", label="Nominal 95%")

    # Improve plot visuals
    ax.set_xlim(true_vars[0], true_vars[-1])
    ax.set_xscale("log")
    ax.set_xlabel("True Variance")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(
        "Coverage of 95% Asymptotic CI for Variance",
        loc="left",
        weight="bold",
    )
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{output_dir}/coverage_plot.svg", bbox_inches="tight")
    plt.close()

    # Length plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    for n_obs, res in results.items():
        lengths = [np.mean(res[var]["lengths"]) for var in true_vars]
        ax.plot(true_vars, lengths, "o-", label=f"{n_obs} observations")
    ax.set_xlim(true_vars[0], true_vars[-1])
    ax.set_xscale("log")
    ax.set_xlabel("True Variance")
    plt.ylabel("Avg. Normalized CI Length")
    plt.title("Lenth of 95% Asymptotic CI for Variance, Divided by Variance")
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{output_dir}/length_plot.svg", bbox_inches="tight")
    plt.close()


def plot_error_kdes(results: dict[int, dict[float, dict]], output_dir: Path) -> None:
    """Plot density of estimation errors for each true variance and sample size.

    A separate figure is exported for each true variance and sample size.

    The expected results dictionary is compatible with the format returned by
    the sim_infrastructure.core.run_simulations() function.

    Args:
        results (dict[int, dict[float, dict]]): dictionary indexed as
            {n_obs: {true_var: {"coverage":..., "length":...}}}.
        output_dir (Path): where to store the resultin gfigures.
    """
    # Make sure output folder exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot separately for each true_var, n_obs
    for n_obs, res in results.items():
        for true_var, data in res.items():
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor(BG_COLOR)

            # Add density of estimation errors
            sns.kdeplot(data["errors"], ax=ax, alpha=0.5, label="MC Errors")

            # Add limit (normal distribution with matching variances)
            error_std = np.sqrt(2 * true_var**2 / n_obs)  
            x = np.linspace(min(data["errors"]), max(data["errors"]), 1000)
            ax.plot(x, norm.pdf(x, 0, error_std), "r-", lw=2, label="Asymptotic Normal")

            # Make plot nicer
            ax.set_title(
                f"Estimation Errors ({n_obs} Observations, True Var={true_var:.4f})",
                weight="bold",
                loc="left",
            )
            ax.set_xlabel("MLE - True Variance")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True)
            fig.savefig(
                f"{output_dir}/errors_n{n_obs}_var_{true_var:.4f}.svg",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()
