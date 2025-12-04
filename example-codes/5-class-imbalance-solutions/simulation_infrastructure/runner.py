"""
Simulation runner for evaluating prediction algorithms via Monte Carlo.

Classes:
    - SimulationRunner: executes simulation scenario, potentially in parallel.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Type

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from simulation_infrastructure.protocols import AlgorithmProtocol, DGPProtocol


class SimulationRunner:
    """Runs Monte Carlo simulations for a given DGP and list of algorithms.

    Attributes:
        dgp_type (Type[DGPProtocol]): class of the Data Generating Process.
        dgp_kwargs (dict[str, Any]): keyword arguments for initializing the DGP.
        algorithm_types (list[Type[AlgorithmProtocol]]): list of algorithm classes.
        algorithm_kwargs_list list[dict[str, Any]]): list of keyword argument
            dictionaries for initializing each algorithm, ordered in the same
            order as algorithm_types
        n_simulations (int): number of Monte Carlo simulations. Defaults to 1000.
        n_workers (int): number of threads for parallel execution. Defaults to 1.
    """

    def __init__(
        self,
        dgp_type: Type[DGPProtocol],
        dgp_kwargs: dict[str, Any],
        algorithm_types: list[Type[AlgorithmProtocol]],
        algorithm_kwargs_list: list[dict[str, Any]],
        n_simulations: int = 1000,
        n_workers: int = 1,
    ):
        self.dgp_type = dgp_type
        self.dgp_kwargs = dgp_kwargs
        self.algorithm_types = algorithm_types
        self.algorithm_kwargs_list = algorithm_kwargs_list
        self.n_simulations = n_simulations
        self.n_workers = n_workers
        self.results = []

    def _run_single_simulation(self, seed: int) -> dict[str, Any]:
        """Run a single Monte Carlo simulation for all algorithms.

        Args:
            seed (int): seed for data sampling
        """
        # Initialize DGP
        dgp = self.dgp_type(**self.dgp_kwargs)
        X_train, X_test, y_train, y_test = dgp.sample(seed=seed)

        sim_results = {}

        # Initialize and fit each algorithm
        for algo_type, algo_kwargs in zip(
            self.algorithm_types, self.algorithm_kwargs_list
        ):
            algo = algo_type(**algo_kwargs)
            algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
            f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

            result_key = dgp.name + " + " + algo.name
            sim_results[result_key] = {
                "n_training": dgp.n_train_samples,
                "first_class_weight": dgp.weights[0],
                "accuracy": accuracy,
                "precision_0": precision_0,
                "recall_0": recall_0,
                "precision_1": precision_1,
                "recall_1": recall_1,
                "f1_0": f1_0,
                "f1_1": f1_1,
            }

        return sim_results

    def run_all(self) -> pd.DataFrame:
        """Run all simulations in parallel and return aggregated results.

        Returns:
            pd.DataFrame: DataFrame with simulation results.
        """
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
