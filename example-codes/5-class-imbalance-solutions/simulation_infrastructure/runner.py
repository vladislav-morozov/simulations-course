"""
Simulation runner for evaluating prediction algorithms via Monte Carlo. 
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any
 
import pandas as pd
from tqdm import tqdm

from simulation_infrastructure.protocols import AlgorithmProtocol, DGPProtocol


class SimulationRunner:
    """Runs Monte Carlo simulations for a given DGP and list of algorithms.
    
    Attributes:
            dgp (DGPProtocol): data generating process that implements a sample()
                method.
            algorithms (list[AlgorithmProtocol]): list of algorithms to evaluate.
                Each algorithm must implement fit() and predict() methods
            n_simulations (int) number of Monte Carlo simulations.
            n_workers (int): number of threads for parallel execution.
    
    """

    def __init__(
        self,
        dgp: DGPProtocol,
        algorithms: list[AlgorithmProtocol],
        n_simulations: int = 1000,
        n_workers: int = 2,
    ): 
        self.dgp = dgp
        self.algorithms = algorithms
        self.n_simulations = n_simulations
        self.n_workers = n_workers
        self.results = []

    def _run_single_simulation(self, seed) -> dict[str, Any]:
        """Run a single Monte Carlo simulation for all algorithms."""
        X_train, X_test, y_train, y_test = self.dgp.sample(seed=seed)
        sim_results = {}

        for algo in self.algorithms:
            algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)

            # Compute metrics
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
            )

            accuracy = accuracy_score(y_test, y_pred)
            precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

            sim_results[algo.name] = {
                "accuracy": accuracy,
                "precision_0": precision_0,
                "recall_0": recall_0,
                "precision_1": precision_1,
                "recall_1": recall_1,
            }

        return sim_results

    def run_all(self) -> pd.DataFrame:
        """Run all simulations in parallel and return aggregated results."""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._run_single_simulation, seed)
                for seed in range(self.n_simulations)
            ]
            for future in tqdm(
                futures, total=self.n_simulations, desc="Running simulations"
            ):
                self.results.append(future.result())

        # Aggregate results into a DataFrame
        df_list = []
        for sim_result in self.results:
            for algo_name, metrics in sim_result.items():
                row = {"algorithm": algo_name, **metrics}
                df_list.append(row)
        return pd.DataFrame(df_list)
