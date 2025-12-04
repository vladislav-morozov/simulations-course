# Simulation 5: Dealing With Unbalanced Classes in Classification

## Overview

Class imbalance is an important problem in classification. Strong imbalance may bias classifiers towards the majority classes, leading to:

- Poor generalizaiton.
- Misleading accuracy.
- Potentially incorrect conclusions.

There are several solutions, including

- Undersampling the majority class.
- Oversampling the minority class.
- Introducing weights in the objective function to give more weight to small classes.

> Do such corrections yield better performance on the minority classes? 

In this simulation we evaluate the effect of class imbalance corrections in a binary classification problem. The metrics considered are  accuracy, recall, $F_1$ score for each algorithm and DGP, printed to the console as a Markdown table.

Overall, we find that techniques like SMOTE and likelihood weight correction improve detection of underrepresented class at the price of more false positives.

For statistical details and code development, see the accompanying lecture: 

- [Evaluating Machine Learning Algorithms](https://vladislav-morozov.github.io/simulations-course/slides/methods/evaluating-ml-algorithms.html) 
 
 
## File Structure


In this simulation, we return to the larger modular format that is is extensible in terms of DGPs and classification approach. Code structure:


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
python -X gil=0 main.py
```  

Results (accuracy, recall, $F_1$ for each algorithm) will be printed to the console.