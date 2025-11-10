# Simulation 2: Joint vs. Multiple Tests

## Overview

This folder contains simulation code for comparing a joint test with a multiple test with corrected critical values.
The driving question is:

> Why do we always use joint test (e.g. Wald) instead of multiple test (e.g. multiple $t$-test with Bonferroni-type corrections) when faced with a joint hypothesis?

Specifically, we consider the following model under normality:

$$
Y_t = \theta_0 + \theta_1 X_i^{(1)} + \theta_2 X_{i}^{(2)} + U_i.
$$

We evaluate power of a joint and a multiple test for testing the null

$$
H_0: \theta_1 =\theta_2 =0.
$$
 
For details and development of code logic, see the following lecture:

- [Evaluating Hypothesis Tests](https://vladislav-morozov.github.io/simulations-course/slides/methods/evaluating-tests.html)

## Tests and DGPs

Each simulation setting is fully described by the triplet (DGP, Test, sample size).

Tests considered are:

- Wald test (joint test).
- Multiple $t$-test with Bonferroni correction (multiple test).


Data generating processes: 

- `dgps.linear.BivariateLinearModel`: normal model, parametrized by common value of $\theta_1=\theta_2$ and correlation between $ X_i^{(1)}$ and $ X_i^{(2)}$.  Sample size: $N=200$
 
The prototypical `BivariateLinearModel` is evaluated on a grid of values for coefficients (to evaluate the power function) and a grid of correlations, yielding approximately $12000$ designs with the preset grid configurations.

## File Structure

``` 
├── README.md
├── requirements.txt
├── dgps
│   ├── __init__.py 
│   └── linear.py 
├── main.py
├── results
│   ├── ...
├── sim_infrastructure
│   ├── __init__.py 
│   ├── orchestrators.py
│   ├── protocols.py 
│   ├── runner.py
│   └── scenarios.py
└── tests
    ├── __init__.py 
    ├── joint.py
    └── multiple.py 
```

The `scenarios.py` file defines the simulation scenarios used.

## How to Reproduce Lecture Results

Specifications used to run code in the lecture slides:
 
- Python 3.14.0t 
- Key packages: `numpy`, `pandas`, `statsmodels` (see `requirements.txt` for list versions)

Install dependencies with:

```sh
python -m pip install -r requirements.txt
```

Run the simulation by executing:

```sh
python -X gil=0 main.py
```  
 