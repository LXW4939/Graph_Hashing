__author__ = 'soloconte'

import numpy as np


class Hashing(object):
    def __init__(self, train_data, bits=32):
        self.train_data = train_data
        self.bits = bits
        self.train_code = np.zeros((self.train_data.shape[0], self.bits))

    def set_bits(self, bits):
        self.bits = bits

    def train(self):
        pass

    def get_hashing_function(self):
        pass

    def hashing_to_code(self, out_of_sample):
        pass