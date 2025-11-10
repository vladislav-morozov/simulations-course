# Simulation 1: Example Data Structure and Bias of OLS-Like Estimators

## Overview

This folder contains simulation code for  

$$
Y_t = \theta_0 + \theta_1 X_i^{(1)} + \theta_2 X_{i}^{(2)} + U_i.
$$
 

Accompanying lectures:

-  

## Estimators and DGPs

Each simulation setting is fully described by the triplet (DGP, Test, sample size).

Tests considered are


Data generating processes:

- 
 

Sample size: $N=200$
 
## File Structure

```
 
```

The `scenarios.py` file defines the simulation scenarios used.

## How to Reproduce Lecture Results

Specifications used to run code in the lecture slides:
 
- Python 3.14.0t 
- Key packages: `numpy`, `pandas`, `statsmodels` (see `requirements.txt` for list versions)

Install dependencies with 

```sh
python -m pip install -r requirements.txt
```

Run the simulation by executing:

```sh
python main.py
```  
 