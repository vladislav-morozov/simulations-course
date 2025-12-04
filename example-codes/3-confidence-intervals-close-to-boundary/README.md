# Simulation 3: Confidence Intervals for Variance when Variance is Close to Parameter Space Boundary

## Overview

Suppose that we are working with a sample from a normal distribution with unknown variance $\sigma^2$. We estimate $\sigma^2$ using the corresponding maximum likelihood estimator $\hat{\sigma}^2$.


Our goal is to conduct inference on $\sigma^2$ by constructing a suitable confidence interval. The easiest way to construct a confidence interval based on $\hat{\sigma}^2$ is using the fact that it's asymptotically normally distributed. 

The key issue is that the normal distribution is symmetric, while the distribution of estimation errors $\hat{\sigma}^2-\sigma^2$ is inherently asymmetric:

- $\hat{\sigma}^2$ can overestimate $\sigma^2$ by any amount.
- But $\hat{\sigma}^2$ cannot *under*estimate $\sigma^2$ by more than $\sigma^2$ (since that would require the absurd result that $\hat{\sigma}^2<0$). 
 

This issue of symmetry leads us to the following questions:


> Does the asymmetry of the finite sample distribution of $\hat{\sigma}^2$ lead to significantly worse performance of asymptotic confidence intervals based on $\hat{\sigma}^2$? Is this effect more pronounced for $\sigma^2\to 0$?

This folder contains simulation code for evaluating coverage and length properties of asymptotic confidence intervals for $\sigma^2$. To understand the effect, the simulation considers a range of values for $\sigma^2$, including smaller values.

The key result is that the value of $\sigma^2$ does not affect the coverage and length properties of confidence intervals. 
 
For details and development of code logic, see the following lecture:

- [Evaluating Confidence Intervals](https://vladislav-morozov.github.io/simulations-course/slides/methods/evaluating-cis.html)
 

## File Structure

This simulation uses a simple function-based code structure in the spirit of the first lecture (see [Good Simulation Code I: Starting With Functions](https://vladislav-morozov.github.io/simulations-course/slides/general-principles/code-basic.html)). 

``` 

├── README.md
├── main.py
├── requirements.txt
└── sim_infrastructure
    ├── __init__.py 
    ├── core.py
    └── plotting.py
```
 
## How to Reproduce Lecture Results


Specifications used to run code in the lecture slides:
 
- Python 3.14.0t 
- Key packages: `numpy` and `pandas` (see `requirements.txt` for versions)

Install dependencies with:

```sh
python -m pip install -r requirements.txt
```

Run the simulation by executing:

```sh
python -X gil=0 main.py
```  