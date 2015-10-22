__author__ = 'soloconte'

import numpy as np


def euclideanDistance(x,y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return ((x-y)**2).sum()
