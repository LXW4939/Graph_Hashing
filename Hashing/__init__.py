__author__ = 'soloconte'


class Hashing(object):
    def __init__(self, train_data, bits=32):
        self.train_data = train_data
        self.bits = bits

    def set_bits(self, bits):
        self.bits = bits

    def train(self):
        pass

    def get_hashing_function(self):
        pass

    def hashing_to_code(self, out_of_sample):
        pass
