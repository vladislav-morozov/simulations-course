"""
Plotting functions for simulation results.

This module contains the following functions:
    - plot_kdes(): plot densities of estimators for each value of N

"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"


def plot_kdes(sim_results: pd.DataFrame, output_dir: Path):
    """Plot KDE plots for each value of N present

    Args:
        sim_results (pd.DataFrame): simulation results in tabular form
        output_dir (Path): where to save the resulting plot
    """

    # Set format variables
    colors = {"No Fixed Effects": "#ffe34d", "With Fixed Effects": "#3c165c"}
    linestyles = {"No Fixed Effects": "-", "With Fixed Effects": "--"}
    bg_color = "whitesmoke"

    # Extract characteristics of data
    n_values = sim_results.n_units.unique()
    est_names = sim_results.model.unique()

    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=len(n_values), figsize=(12, 3.6))
    fig.patch.set_facecolor(bg_color)

    # Add KDEs for each value of N
    for ax_id, ax in enumerate(axs):
        # Extract data corresponding to current N
        sim_results_for_n = sim_results.loc[
            sim_results["n_units"] == n_values[ax_id], ["model", "coef_est"]
        ]

        # Background color gradient
        ax.set_facecolor(bg_color)

        for y in est_names:
            a = sns.kdeplot(
                sim_results_for_n.loc[
                    sim_results_for_n["model"] == y, "coef_est"
                ].squeeze(),
                ax=ax,
                bw_adjust=2,
                label=y,
                color=colors[y],
                fill=True,
                alpha=0.5,
                linestyle=linestyles[y],
            )

        # Add a line at y = 1
        ax.axvline(-0.25, color="black", alpha=0.5, linestyle=":")

        # Title and axis labels
        ax.set_title(f"N={n_values[ax_id]}", loc="left", color="black", fontsize=14)

        # Adjust ticks
        ax.tick_params(axis="x", colors="black", labelsize=9)
        if ax_id == 0:
            ax.tick_params(axis="y", colors="black", labelsize=12)
            ax.set_ylabel("Density", fontsize=12, color="black")
        else:
            ax.tick_params(left="False")
            ax.get_yaxis().set_visible(False)

        ax.set_xlabel(" ", fontsize=12, color="black")
        ax.set_xlim(-0.6, 0.75)

        # Generate one legend
        if ax_id == len(n_values) - 1:
            ax.legend(
                bbox_to_anchor=(0.3, 1),
                fontsize=12,
                edgecolor="black",
                labelcolor="black",
            )

    fig.suptitle(
        "Distributions of OLS and FE Estimators",
        x=0.375,
        y=1.1,
        color="black",
        fontsize=20,
        weight="bold",
    )
    # Export as SVG
    fig.savefig(output_dir / "kde_results.svg", bbox_inches="tight")
    plt.close()
