__author__ = 'soloconte'

from Hashing import Hashing
from Utility import AnchorPoint, Distance
import numpy as np
from scipy import sparse


class AnchorGraphHashing(Hashing):
    def __init__(self, train_data, bits=32, anchor_num=100, nearest_num=2, sigma=0):
        super(AnchorGraphHashing, self).__init__(train_data, bits)
        self.anchorPoints = []
        self.anchor_num = anchor_num
        self.nearest_num = 2
        self.distance = []
        self.sigma = sigma
        self.matZ = sparse.dok_matrix((self.train_data.shape[0], self.anchor_num), float)

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

    def setZ(self):
        if self.sigma == 0:
            pass
        for i in xrange(len(self.distance)):
            for j, dis in self.distance[i]:
                self.matZ[(i, j)] = np.exp(-dis / (self.sigma ** 2))
        self.matZ = sparse.csr_matrix(self.matZ / self.matZ.sum(1))


    def setW(self):
        matA = np.diag(self.matZ.sum(0))
        matA = np.matrix(matA**0.5).I
        matM = matA * self.matZ.T * self.matZ * matA


    def train(self):
        self.selectNearestAnchor()
        self.setZ()
        self.setW()

    def get_hashing_function(self):
        pass

    def hashing_to_code(self, out_of_sample):
        pass



