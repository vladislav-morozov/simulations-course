"""
Entry point for running simulation on classification with class imbalance.

Overall goal of simulation: evaluate effect of various techniques for dealing
with unbalanced classes in binary classification problems.

The code compares correction techniques in terms of precision, recall, and the
F_1 score. Techniques considered:
    - Not doing anything.
    - SMOTE (synthetic oversampling).
    - Introducing class weights in the criterion function.

Usage:
    python -X gil=0 main.py

Output:
    Console printout of precision, recall, and F1 scores
"""

import pandas as pd

from simulation_infrastructure.orchestrator import SimulationOrchestrator
from simulation_infrastructure.scenarios import scenarios


def main():
    # Create and run the orchestrator
    orchestrator = SimulationOrchestrator(scenarios, n_workers=4)
    orchestrator.run_all()
    combined_results = pd.concat(orchestrator.summary_results.values())

    # Print key results as a markdown table
    print(
        combined_results.groupby(by=["algorithm", "n_training", "first_class_weight"])[
            ["precision_1", "recall_1", "f1_1"]
        ]
        .mean()
        .round(3)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
