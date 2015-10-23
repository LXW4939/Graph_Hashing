__author__ = 'soloconte'

import Hashing
from Preprocess import AGHPreprocess
import numpy as np
from scipy import sparse, linalg

class AnchorGraphHashing(Hashing.Hashing):
    def __init__(self, train_data, bits=32, anchor_num=300, nearest_num=2, sigma=0):
        super(AnchorGraphHashing, self).__init__(train_data, bits)
        self.hashMat = []
        self.anchor_num = anchor_num
        self.nearest_num = nearest_num
        self.sigma = sigma
        self.preprocessor = AGHPreprocess.AGHPreprocess(train_data, self.anchor_num, self.nearest_num, self.sigma)

    def train(self):
        print "Start Training"
        matZ = self.preprocessor.process(self.train_data)
        print "Matrix Z Calculated"
        matA = np.diag(matZ.toarray().sum(0))
        matA = np.matrix(matA**0.5).I
        print "size of matA : %d * %d" % matA.shape
        print "size of matZ : %d * %d" % matZ.shape
        matM = matA * matZ.T * matZ * matA
        eigVals, eigVecs = linalg.eig(matM)
        print "Eigen Calculation Finished"
        minIndex = eigVals.argsort()[-self.bits-1:-1]
        eigVals = eigVals[minIndex]
        eigVecs = eigVecs[minIndex]
        eigVals = np.diag(eigVals)
        eigVals = np.matrix(eigVals ** 0.5).I
        self.hashMat = matA * np.matrix(eigVecs.T) * eigVals
        print "Training Finished"

    def get_hashing_function(self):
        return self.hashMat

    def hashing_to_code(self, out_of_sample):
        test_matZ = self.preprocessor.process(out_of_sample)
        binary_code = np.zeros((out_of_sample.shape[0], self.bits), int)
        code = test_matZ * self.hashMat
        binary_code[code > 0] = 1
        print "Test Code Generated"
        return binary_code





