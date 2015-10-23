__author__ = 'soloconte'

import Hashing
from Preprocess import AGHPreprocess
import numpy as np
from scipy import linalg


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
        mat_z = self.preprocessor.process(self.train_data)
        print "Matrix Z Calculated"
        mat_a = np.diag(mat_z.toarray().sum(0))
        mat_a = np.matrix(mat_a**0.5).I
        print "size of matA : %d * %d" % mat_a.shape
        print "size of matZ : %d * %d" % mat_z.shape
        mat_m = mat_a * mat_z.T * mat_z * mat_a
        eig_vals, eig_vecs = linalg.eig(mat_m)
        print "Eigen Calculation Finished"
        min_index = eig_vals.argsort()[-self.bits-1:-1]
        eig_vals = eig_vals[min_index]
        eig_vecs = eig_vecs[min_index]
        eig_vals = np.diag(eig_vals)
        eig_vals = np.matrix(eig_vals ** 0.5).I
        self.hashMat = mat_a * np.matrix(eig_vecs.T) * eig_vals
        print "Training Finished"

    def get_hashing_function(self):
        return self.hashMat

    def hashing_to_code(self, out_of_sample):
        test_matz = self.preprocessor.process(out_of_sample)
        binary_code = np.zeros((out_of_sample.shape[0], self.bits), int)
        code = test_matz * self.hashMat
        binary_code[code > 0] = 1
        print "Test Code Generated"
        return binary_code
