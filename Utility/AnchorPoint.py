__author__ = 'soloconte'

import numpy as np
from Utility import Distance

class AnchorPoint(object):
    def __init__(self, train_data, anchor_num=100):
        self.train_data = train_data
        self.anchor_num = anchor_num
        self.anchor_points = self.train_data[:anchor_num]
        self.index = np.zeros(self.train_data.shape[0], int)

    def split(self, Measurement=Distance.euclideanDistance):
        for i in xrange(self.train_data.shape[0]):
            tdata = self.train_data[i]
            distance = np.asarray([Measurement(tdata, adata) for adata in self.anchor_points])
            self.index[i]= distance.argmin()

    def update(self):
        new_points = np.zeros(self.anchor_points.shape, float)
        cluster_num = np.zeros(new_points.shape[0], int)
        for i in xrange(self.train_data.shape[0]):
            new_points[self.index[i]] += self.train_data[i]
            cluster_num[self.index[i]] += 1
        self.anchor_points = (new_points.T/cluster_num).T

    def setAnchorPoints(self, iter_time=10):
        for i in xrange(iter_time):
            self.split()
            self.update()
        return self

    def getAnchorPoints(self):
        import pickle
        data_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/anchors300", "w")
        pickle.dump(self.anchor_points, data_file)
        data_file.close()
        return self.anchor_points