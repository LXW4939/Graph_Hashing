__author__ = 'soloconte'

import numpy as np
from Utility import SelectNearest

class Recommendation(object):
    def __init__(self, data, hashing):
        self.data = data
        self.hashing = hashing
        self.hashing.train()
        self.train_code = np.array([])
        self.test_code = np.array([])
        self.nearest = []
        self.result = []

    def generate_code(self):
        self.train_code =self.hashing.hashing_to_code(self.data.train_data)
        self.test_code = self.hashing.hashing_to_code(self.data.test_data)
        print "Code Generated"

    def NN_search(self, selectNearest, similar_user_number=10):
        for test_user in self.test_code:
            my_neighbour = selectNearest(test_user, self.train_code, similar_user_number)
            self.nearest.append(my_neighbour)
        print "Neighbours Found"

    def recommend(self, selectNearest=SelectNearest.selectRadius, similar_user_number=10, numberOfRec=10):
        self.generate_code()
        self.NN_search(selectNearest)
        #for neighbours in self.nearest:
            #score = np.zeros((1, self.data.test_data.shape[1]))
            #for sim, user in neighbours:
                #score += sim * self.data.train_data[user]
            #self.result.append(score.argsort()[-numberOfRec:])
