__author__ = 'soloconte'

import numpy as np


def GaussianRBF(distance, sigma=1):
    return np.exp(-distance / (sigma**2))
