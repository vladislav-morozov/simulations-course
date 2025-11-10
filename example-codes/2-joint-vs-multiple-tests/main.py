"""
Entry point for running simulations for comparing linear

Overall goal of simulation: evaluate bias of OLS-like estimators in simple model
    y = b0 + b1 * x + u
Scenarios consider: several estimators, static vs. dynamic DGPs, various sample
sizes.

Usage:
    python main.py

Output:
    A pandas Series of bias results for each scenario, printed to the console.
"""

from pathlib import Path

import pandas as pd

from sim_infrastructure.orchestrators import SimulationOrchestratorParallel
from sim_infrastructure.results_processor import ResultProcessor
from sim_infrastructure.scenarios import scenarios

SIM_RESULTS_PATH = Path() / "results" / "sim_results.csv"
PLOT_FOLDER = Path() / "results" / "plots"


if __name__ == "__main__":
    # Create and execute simulations
    orchestrator = SimulationOrchestratorParallel(scenarios)
    orchestrator.run_all()

    # Export results
    pd.DataFrame(orchestrator.summary_results).to_csv("results/sim_results.csv")

    # Eexport plots
    results_processor = ResultProcessor(SIM_RESULTS_PATH, PLOT_FOLDER)
    results_processor.export_all_plots()
