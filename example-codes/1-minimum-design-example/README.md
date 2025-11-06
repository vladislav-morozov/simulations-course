# Simulation 1: Example Data Structure and Bias of OLS-Like Estimators

## Overview

This folder contains simulation code for evaluating the bias of several OLS-like estimators in the simple time series model:

$$
Y_t = \beta_0 + \beta_1 X_t + U_t, \quad t=1, \dots, T.
$$

Key technical goal of this simulation: demonstrate a modular framework for organizing and running Monte Carlo simulations.

Accompanying lectures:

- [Good Simulation Code I: Starting with Functions](https://vladislav-morozov.github.io/simulations-course/slides/general-principles/code-basic.html)
- [Good Simulation Code II: Modular Approachs](https://vladislav-morozov.github.io/simulations-course/slides/general-principles/code-modular.html)
- [Good Simulation Code III: Organization](https://vladislav-morozov.github.io/simulations-course/slides/general-principles/code-organization.html)
- [Good Simulation Code IV: Orchestration](https://vladislav-morozov.github.io/simulations-course/slides/general-principles/code-orchestration.html)

## Estimators and DGPs

Each simulation setting is fully described by the triplet (DGP, Estimator, sample size).

Estimators considered are (along with their class names):

- OLS (`SimpleOLS`)
- Ridge (`SimpleRidge`)
- Lasso (`LassoWrapper`)

Data generating processes:

- Static process (`StaticNormalDGP`): $X_t$ and $U_t$ standard normal, mutually independent and independent over time, $\beta_0=0, \beta_1=0.5$.
- Dynamic process (`DynamicNormalDGP`): $X_t= Y_{t-1}$; $U_t$ standard normal. $\beta_0=0$, three values considered for $\beta_1\in \lbrace 0, 0.5, 0.95\rbrace$.
 
Two sample sizes: $T=50, T=200$.
 
## File Structure

```
project/
├── dgps/
│   ├── __init__.py
│   ├── static.py       # StaticNormalDGP
│   └── dynamic.py      # DynamicNormalDGP
├── estimators/
│   ├── __init__.py
│   ├── ols-like.py     # SimpleOLS, SimpleRidge, LassoWrapper
├── main.py             # Main script: entry point
├── orchestrator.py     # SimulationOrchestrator
├── protocols.py        # DGPProtocol, EstimatorProtocol
├── runner.py           # SimulationRunner
└── scenarios.py        # scenarios and SimulationScenario
```

The `scenarios.py` file defines the simulation scenarios used.

## How to Reproduce Lecture Results

Specifications used to run code in the lecture slides:
 
- Python 3.14.0t 
- Key packages: `numpy`, `pandas`, `scikit-learn` (see `requirements.txt` for list versions)

Install dependencies with 

```sh
python -m pip install -r requirements.txt
```

Run the simulation by executing:

```sh
python main.py
```  

Results (bias for each scenario) will be printed to the console.