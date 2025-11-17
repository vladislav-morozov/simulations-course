# Simulation 4: When Adding Fixed Effects Worsens Bias

## Overview


> Is it always true that adding more fixed effects means that the point estimates are closer to the true average coefficients?

This folder contains simulation code for finding and simulating a scenario where a fixed effects estimator is more biased for the average effect than a simple OLS estimator.

Specifically, we consider the following potential outcomes model under normality:

$$
Y_{it}^x = \alpha_i + \beta_i x + U_{it}
$$ 

This setting is more realistic than a simple random intercept model, as it allows for treatment effect heterogeneity. 

The simulation in this folder showcases a scenario in which  adding fixed effects to estimation (eliminating $\alpha_i$) yields an estimator that almost always has a different sign from $\mathbb{E}[\beta_i]$. In the same scenario, a simple OLS estimator is generally very close to $\mathbb{E}[\beta_i]$.
 
For details and development of code logic, see the following lecture:

- [Evaluating Causal Estimators](https://vladislav-morozov.github.io/simulations-course/slides/methods/evaluating-causal-estimators.html)
 

## File Structure

``` 
├── dgp
│   ├── constants.py
│   ├── data_generation.py
│   ├── __init__.py
│   ├── moment_info.py 
├── exp.ipynb
├── gmm
│   ├── __init__.py 
│   └── solver.py
├── main.py
├── README.md 
│   └── ...
└── simulation_infrastructure
    ├── constants.py
    ├── __init__.py
    ├── plotting.py 
    └── runner_functions.py

```
 
## How to Reproduce Lecture Results

Specifications used to run code in the lecture slides:
 
- Python 3.12.8 (note different version relative to other simulations!) 
- Key packages is `pyfixest` (see `requirements.txt` for list versions)

Install dependencies with:

```sh
python -m pip install -r requirements.txt
```

Run the simulation by executing:

```sh
python -X gil=0 main.py
```  
 