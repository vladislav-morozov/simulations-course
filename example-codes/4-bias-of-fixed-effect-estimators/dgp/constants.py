"""
This module contains the constants used for specifying the DGP

Constants:
- BETA_MEAN (float): Mean value for the slope used in the simulation.
- N_VALUES (numpy.ndarray): Array of values representing different sample sizes 
    for the simulation.
"""

import numpy as np

BETA_MEAN = -0.25
N_VALUES = np.array([100, 200, 500])
