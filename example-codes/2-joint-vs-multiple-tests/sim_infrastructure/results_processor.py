"""
Module for processing and visualizing simulation results.

Classes:
    ResultsProcessor: exports figures that appear in the slides.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResultsProcessor:
    """Visualizes simulation results for comparing Wald vs. Bonferroni t test.

    Attributes:
        input_path (Path): Path to the input CSV file containing simulation results.
        output_path (Path): Path to the directory where plots will be saved.
    """

    def __init__(self, input_path: Path, output_path: Path) -> None:
        # Load in and process simulation results
        self._prepare_power_data(input_path)

        # Prepare for outputs
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set colors
        self.bg_color = "whitesmoke"

    def export_all_plots(self) -> None:
        """Exports all plots to the output directory.

        This method calls all specific plot-exporting methods.
        """
        self.export_power_surfaces()
        self.export_power_differences()

    def export_power_surfaces(self) -> None:
        """Exports a power plot from the simulation results.
        
        The figure has two subplots, one for each test considered. Each subplot
        depicts the power surface as a function of the DGP parameters.
        """
        # Create meshgrid
        c_mesh, rho_mesh = np.meshgrid(
            self.coef_val_range, self.corr_range, indexing="ij"
        )

        # Create figure and add surface plots
        fig = plt.figure(figsize=(16, 6))
        fig.patch.set_facecolor(self.bg_color)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(
            c_mesh,
            -rho_mesh,
            self.wald_results,
            edgecolor="peru",
            lw=0.5,
            alpha=0.3,
        )
        ax1.contour(
            c_mesh,
            rho_mesh,
            self.wald_results,
            zdir="y",
            offset=-self.corr_range.min(),
            cmap="coolwarm",
        )

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(
            c_mesh,
            -rho_mesh,
            self.bonf_results,
            edgecolor="peru",
            lw=0.5,
            alpha=0.34,
        )
        ax2.contour(
            c_mesh,
            rho_mesh,
            self.bonf_results,
            zdir="y",
            offset=-self.corr_range.min(),
            cmap="coolwarm",
        )

        axs = [ax1, ax2]

        # Set basic figure colors
        for ax in axs:
            ax.view_init(elev=30, azim=120, roll=0)
            # Set axes
            ax.set(
                xlim=(self.coef_val_range.min() - 0.1, self.coef_val_range.max() + 0.1),
                ylim=(self.corr_range.min(), self.corr_range.max()),
                zlim=(-0.1, 1),
                facecolor=self.bg_color,
            )

            ax.set_xlabel("c: Value of $\\theta_1=\\theta_2$", color="black")
            ax.set_ylabel(
                "Correlation between $\\hat{\\theta_2}_1$ and $\\hat{\\theta_2}_2$",
                color="black",
            )

        ax1.set_title("Joint: Wald", color="black", loc="left", fontsize=16)
        ax2.set_title(
            "Multiple $t$ with Bonferroni Adjustment",
            color="black",
            loc="left",
            fontsize=16,
        )

        # Adjust layout to add padding for the text
        fig.subplots_adjust(top=0.85, wspace=0.04, bottom=-0.14)

        # Add title section
        fig.text(
            0.125,
            1.05,
            "Power Functions of Wald and Adjusted Multiple $t$-Tests",
            ha="left",  # Horizontal alignment
            va="top",  # Vertical alignment
            fontsize=18,
            color="black",
            weight="bold",
        )

        # Export as SVG
        plt.savefig(self.output_path / "power_surfaces.svg", bbox_inches="tight")

    def export_power_differences(self) -> None:
        """Export a plot with power differences between Wald and multiple tests.

        The figure shows the difference in power between the two tests 
        as a function of the DGP parameters.
        """
        # Create meshgrid
        c_mesh, rho_mesh = np.meshgrid(
            self.coef_val_range, self.corr_range, indexing="ij"
        )

        # Create figure and add surface plots
        fig = plt.figure(figsize=(11, 6))
        fig.patch.set_facecolor(self.bg_color)

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(
            c_mesh,
            -rho_mesh,
            self.wald_results - self.bonf_results,
            edgecolor="peru",
            lw=0.5,
            alpha=0.3,
        )
        ax.contour(
            c_mesh,
            rho_mesh,
            self.wald_results - self.bonf_results,
            zdir="y",
            offset=-self.corr_range.min(),
            cmap="coolwarm",
        )

        ax.view_init(elev=30, azim=120, roll=0)
        # Set axes
        ax.set(
            xlim=(self.coef_val_range.min() - 0.1, self.coef_val_range.max() + 0.1),
            ylim=(self.corr_range.min(), self.corr_range.max()),
            zlim=(-0.1, 1),
            facecolor=self.bg_color,
        )

        ax.set_xlabel("c: Value of $\\theta_1=\\theta_2$", color="black")
        ax.set_ylabel(
            "Correlation between $\\hat{\\theta_2}_1$ and $\\hat{\\theta_2}_2$",
            color="black",
        )

        ax.set_title(
            "Difference in Power Between Wald and Multiple $t$-test",
            loc="left",
            weight="bold",
        )
        fig.subplots_adjust(top=0.85, wspace=0.04, bottom=-0.14)

        # Export as SVG
        plt.savefig(self.output_path / "power_diff.svg", bbox_inches="tight")

    def _prepare_power_data(self, input_path: Path) -> None:
        """Load and prepare arrays of power functions for tests considered.

        Args:
            input_path (Path): path to table with simulation results
        """
        sim_results = pd.read_csv(input_path)

        # Extract ranges of DGP parameters used
        self.coef_val_range = sim_results["common_coef_val"].unique()
        self.corr_range = sim_results["rho"].unique()

        # Process and reshape power arrays
        self.wald_results = (
            sim_results.loc[sim_results.loc[:, "test"] == "Wald"]
            .groupby(["common_coef_val", "rho"])["power"]
            .mean()
            .unstack()
        )
        self.bonf_results = (
            sim_results.loc[sim_results.loc[:, "test"] == "Bonferroni t"]
            .groupby(["common_coef_val", "rho"])["power"]
            .mean()
            .unstack()
        )
