"""
Entry point for running simulations for comparing power of a joint Wald test
vs. a multiple t-test. 

This module contains scenarios for comparing power functions of tests under model:
    Y = theta0  + theta1*x1 + theta2*x2 + u
The null being tested is
    H0: theta1=theta2=0

Scenarios considered: normal DGP with varying values of coefficients and varying
values of correlation between x1 and x2.
    

Usage:
    python -X gil=0 main.py

Output:
    Power surface comparison plots saved under PLOT_FOLDER
"""

from pathlib import Path

import pandas as pd

from sim_infrastructure.orchestrators import SimulationOrchestratorParallel
from sim_infrastructure.results_processor import ResultsProcessor
from sim_infrastructure.scenarios import scenarios

SIM_RESULTS_PATH = Path() / "results" / "sim_results.csv"
PLOT_FOLDER = Path() / "results" / "plots"


if __name__ == "__main__":
    # Create and execute simulations
    orchestrator = SimulationOrchestratorParallel(scenarios)
    orchestrator.run_all()

    # Export results
    pd.DataFrame(orchestrator.summary_results).to_csv(SIM_RESULTS_PATH)

    # Export plots
    results_processor = ResultsProcessor(SIM_RESULTS_PATH, PLOT_FOLDER)
    results_processor.export_all_plots()
