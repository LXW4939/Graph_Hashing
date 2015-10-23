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

    def select_nearest_anchor(self, data, measurement=Distance.euclidean_distance, iter_time=10):
        if self.anchorPoints == []:
            if False:
                self.anchorPoints = AnchorPoint.AnchorPoint(self.train_data, self.anchor_num). \
                    set_anchor_points(iter_time).get_anchor_points()

            # ###############  temp code  ##############
            import pickle
            data_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/anchors300")
            self.anchorPoints = pickle.load(data_file)
            data_file.close()
            # ##########################################

        n, m = data.shape[0], self.anchor_num
        self.distance = []
        for i in xrange(n):
            distance_i = []
            for j in xrange(m):
                distance_i.append(measurement(data[i], self.anchorPoints[j]))
            min_index = np.asarray(distance_i).argsort()[:self.nearest_num]
            self.distance.append([(index, distance_i[index]) for index in min_index])

    def calculate_z(self, data):
        mat_z = sparse.dok_matrix((data.shape[0], self.anchor_num), float)
        if self.sigma == 0:
            max_dis = [anchors[-1][-1] ** 0.5 for anchors in self.distance]
            self.sigma = np.asarray(max_dis).mean()
        for i in xrange(len(self.distance)):
            for j, dis in self.distance[i]:
                mat_z[(i, j)] = np.exp(-dis / (self.sigma ** 2))
        mat_z = sparse.csr_matrix(mat_z / mat_z.sum(1))
        return mat_z

    def process(self, data):
        self.select_nearest_anchor(data)
        print "Nearest Anchors Selected"
        return self.calculate_z(data)
