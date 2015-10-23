__author__ = 'soloconte'

from Utility import Distance, Kernels
import numpy as np


def select_topk(test_code, train_codes, num_of_result=10, measure=Distance.euclidean_distance,
                kernel=Kernels.GaussianRBF):
    distances = []
    for code in train_codes:
        distances.append(measure(test_code, code))
    distances = np.asarray(distances)
    min_index = distances.argsort()[:num_of_result]
    result = [(kernel(distances[index]), index) for index in min_index]
    return result


def select_radius(test_code, train_codes, radius_threshold=2, measure=Distance.euclidean_distance,
                  kernel=Kernels.GaussianRBF):
    distances = []
    for code in train_codes:
        distances.append(measure(test_code, code))
    result = [(kernel(distances[index]), index) for index in xrange(len(distances))
              if distances[index] <= radius_threshold]
    return result
