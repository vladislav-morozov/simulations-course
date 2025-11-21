# Simulation 5: Dealing With Unbalanced Classes in Classification

## Overview

This folder contains simulation code for evaluating the effect of class imbalance corrections in a binary classification problem. 
The code is extensible in terms of DGPs and classification approach. Output: accuracy, recall, $F_1$ score for each algorithm and DGP, printed to the console as a markdown table.

Accompanying lectures:

- [Evaluating Machine Learning Algorithms](https://vladislav-morozov.github.io/simulations-course/slides/methods/evaluating-ml-algorithms.html) 
 
 
## File Structure

```
├── README.md
├── algorithms
│   ├── __init__.py 
│   └── logistic.py
├── dgps
│   ├── __init__.py 
│   └── sklearn_based.py
├── main.py
└── simulation_infrastructure
    ├── __init__.py 
    ├── orchestrator.py
    ├── protocols.py
    ├── runner.py
    └── scenarios.py
```
 

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

Results (accuracy, recall, $F_1$ for each algorithm) will be printed to the console.