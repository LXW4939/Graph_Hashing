__author__ = 'soloconte'

import Hashing


class DiscreteGraphHashing(Hashing.Hashing):
    def __init__(self, train_data, bits=32):
        super(DiscreteGraphHashing, self).__init__(train_data, bits)

    def train(self):
        super(DiscreteGraphHashing, self).train()

    def get_hashing_function(self):
        super(DiscreteGraphHashing, self).get_hashing_function()

    def hashing_to_code(self, out_of_sample):
        super(DiscreteGraphHashing, self).hashing_to_code(out_of_sample)
