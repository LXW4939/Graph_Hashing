__author__ = 'soloconte'

from Preprocess import Preprocess
from scipy import sparse
from Utility import AnchorPoint, Distance
import numpy as np


class AGHPreprocess(Preprocess):
    def __init__(self, train_data, anchor_num=300, nearest_num=2, sigma=0):
        super(AGHPreprocess, self).__init__(train_data)
        self.anchor_num = anchor_num
        self.nearest_num = nearest_num
        self.sigma = sigma
        self.anchorPoints = []
        self.distance = []

    def selectNearestAnchor(self, data, Measurement=Distance.euclideanDistance, iter_time=10):
        if self.anchorPoints == []:
            self.anchorPoints = AnchorPoint.AnchorPoint(self.train_data, self.anchor_num).   \
                                setAnchorPoints(iter_time).getAnchorPoints()
        # import pickle
        #
        # data_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/anchors300")
        # self.anchorPoints = pickle.load(data_file)
        # data_file.close()
        n, m = data.shape[0], self.anchor_num
        self.distance = []
        for i in xrange(n):
            distance_i = []
            for j in xrange(m):
                distance_i.append(Measurement(data[i], self.anchorPoints[j]))
            minIndex = np.asarray(distance_i).argsort()[:self.nearest_num]
            self.distance.append([(index, distance_i[index]) for index in minIndex])

    def calculateZ(self, data):
        matZ = sparse.dok_matrix((data.shape[0], self.anchor_num), float)
        if self.sigma == 0:
            maxDis = [anchors[-1][-1] ** 0.5 for anchors in self.distance]
            self.sigma = np.asarray(maxDis).mean()
        for i in xrange(len(self.distance)):
            for j, dis in self.distance[i]:
                matZ[(i, j)] = np.exp(-dis / (self.sigma ** 2))
        matZ = sparse.csr_matrix(matZ / matZ.sum(1))
        return matZ

    def process(self, data):
        self.selectNearestAnchor(data)
        print "Nearest Anchors Selected"
        return self.calculateZ(data)

