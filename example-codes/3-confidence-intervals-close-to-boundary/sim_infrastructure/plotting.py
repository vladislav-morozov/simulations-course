from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

def plot_coverage_and_lengths(
    results: dict[int, dict[float, dict]], output_dir: Path
) -> None:
    """Plot coverage and normalized interval lengths vs. true variance."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    true_vars = list(next(iter(results.values())).keys())

    # Coverage plot
    plt.figure(figsize=(10, 6))
    for n_obs, res in results.items():
        coverages = [res[var]["coverage"] for var in true_vars]
        plt.plot(true_vars, coverages, "o-", label=f"n_obs={n_obs}")
    plt.axhline(0.95, color="gray", linestyle="--", label="Nominal 95%")
    plt.xscale("log")
    plt.xlabel("True Variance")
    plt.ylabel("Empirical Coverage")
    plt.title("Coverage vs. True Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/coverage_plot.svg",  bbox_inches="tight")
    plt.close()

    # Normalized length plot
    plt.figure(figsize=(10, 6))
    for n_obs, res in results.items():
        lengths = [np.mean(res[var]["lengths"]) for var in true_vars]
        plt.plot(true_vars, lengths, "o-", label=f"n_obs={n_obs}")
    plt.xscale("log")
    plt.xlabel("True Variance")
    plt.ylabel("Avg. Normalized CI Length")
    plt.title("Normalized CI Length vs. True Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/length_plot.png", bbox_inches="tight")
    plt.close()


def plot_error_kdes(
    results: dict[int, dict[float, dict]], output_dir: Path
) -> None:
    """Plot KDE of estimation errors for each true variance and sample size."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for n_obs, res in results.items():
        for true_var, data in res.items():
            plt.figure(figsize=(10, 6))
            error_std = np.sqrt(2 * true_var**2 / n_obs)  # Asymptotic std of MLE
            x = np.linspace(min(data["errors"]), max(data["errors"]), 1000)
            plt.hist(
                data["errors"], bins=30, density=True, alpha=0.5, label="MC Errors"
            )
            plt.plot(
                x, norm.pdf(x, 0, error_std), "r-", lw=2, label="Asymptotic Normal"
            )
            plt.title(f"Estimation Errors (n_obs={n_obs}, True Var={true_var:.4f})")
            plt.xlabel("MLE - True Variance")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                f"{output_dir}/errors_n{n_obs}_var_{true_var:.4f}.svg",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()
