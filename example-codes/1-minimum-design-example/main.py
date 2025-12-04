"""
Entry point for running simulation on bias of simple linear model.

Overall goal of simulation: evaluate bias of OLS-like estimators in simple model
    Y = b0 + b1 * X + U

Scenarios considered differ in three respects:

- Estimators (e.g. OLS, Ridge, Lasso)
- DGP for Y and X:  (e.g. static vs. dynamic processes)
- Sample sizes.

Usage:
    python main.py

Output:
    A pandas Series of bias results for each scenario, printed to the console.
"""

import pandas as pd

from orchestrator import SimulationOrchestrator
from scenarios import scenarios

if __name__ == "__main__":
    # Create and execute simulations
    orchestrator = SimulationOrchestrator(scenarios)
    orchestrator.run_all()

    # Results logic (print or export as pd.Series)
    print(pd.Series(orchestrator.summary_results))
