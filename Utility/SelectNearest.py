__author__ = 'soloconte'

from Utility import Distance, Dis2Sim
import numpy as np


def selectTopK(testCode, trainCodes, numOfResult=10, measure=Distance.euclideanDistance):
    distances = []
    for code in trainCodes:
        distances.append(measure(testCode,code))
    distances = np.asarray(distances)
    minIndex= distances.argsort()[:numOfResult]
    result = [(Dis2Sim.dis2Sim(distances[index]), index) for index in minIndex]
    return result


def selectRadius(testCode, trainCodes, radiusThreshold=2, measure=Distance.euclideanDistance):
    distances = []
    for code in trainCodes:
        distances.append(measure(testCode,code))
    result = [(Dis2Sim.dis2Sim(distances[index]), index) for index in xrange(len(distances)) \
              if distances[index] <= radiusThreshold]
    return result