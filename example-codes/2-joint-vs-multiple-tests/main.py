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

import pandas as pd

from sim_infrastructure.orchestrator import SimulationOrchestrator 
from sim_infrastructure.scenarios import scenarios               

if __name__ == "__main__": 

    # Create and execute simulations 
    orchestrator = SimulationOrchestrator(scenarios)                         
    orchestrator.run_all()

    # Handling results
    pd.DataFrame(orchestrator.summary_results).to_csv("results/sim_results.csv")