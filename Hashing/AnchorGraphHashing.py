__author__ = 'soloconte'

from Hashing import Hashing
from Utility import AnchorPoint, Distance
import numpy as np
from scipy import sparse, linalg

class AnchorGraphHashing(Hashing):
    def __init__(self, train_data, bits=32, anchor_num=100, nearest_num=2, sigma=0):
        super(AnchorGraphHashing, self).__init__(train_data, bits)
        self.anchorPoints = []
        self.anchor_num = anchor_num
        self.nearest_num = 2
        self.distance = []
        self.sigma = sigma
        self.matZ = sparse.dok_matrix((self.train_data.shape[0], self.anchor_num), float)
        self.hashMat = np.zeros((self.anchor_num, self.bits))

    def selectNearestAnchor(self, Measurement=Distance.euclideanDistance, iter_time=10):
        self.anchorPoints = AnchorPoint.AnchorPoint(self.train_data, self.anchor_num).   \
                            setAnchorPoints(iter_time).getAnchorPoints()
        n, m = self.matZ.shape
        for i in xrange(n):
            distance_i = []
            for j in xrange(m):
                distance_i.append(Measurement(self.train_data[i], self.anchorPoints[j]))
            minIndex = np.asarray(distance_i).argsort()[:self.nearest_num]
            self.distance.append([(index, distance_i[index]) for index in minIndex])

    def calculateZ(self):
        if self.sigma == 0:
            maxDis = [anchors[-1][-1]**0.5  for anchors in self.distance]
            self.sigma = np.asarray(maxDis).mean()
        for i in xrange(len(self.distance)):
            for j, dis in self.distance[i]:
                self.matZ[(i, j)] = np.exp(-dis / (self.sigma ** 2))
        self.matZ = sparse.csr_matrix(self.matZ / self.matZ.sum(1))


    def trainHashMat(self):
        matA = np.diag(self.matZ.sum(0))
        matA = np.matrix(matA**0.5).I
        matM = matA * self.matZ.T * self.matZ * matA
        eigVals, eigVecs = linalg.eig(matM)
        minIndex = eigVals.argsort()[-self.bits-1:-1]
        eigVals = eigVals[minIndex]
        eigVecs = eigVecs[minIndex]
        eigVals = np.diag(eigVals)
        eigVals = np.matrix(eigVals ** 0.5).I
        self.hashMat = matA * np.matrix(eigVecs.T) * eigVals

    def train(self):
        self.selectNearestAnchor()
        self.calculateZ()
        self.trainHashMat()

    def get_hashing_function(self):
        return self.hashMat

    def hashing_to_code(self, out_of_sample):
        binary_code = np.zeros((out_of_sample.shape[0], self.bits), int)
        code = np.matrix(out_of_sample) * self.hashMat
        binary_code[code > 0] = 1
        return binary_code




