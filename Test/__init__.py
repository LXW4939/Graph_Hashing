__author__ = 'soloconte'

import numpy as np


class Test(object):
    def __init__(self, data, hashing):
        self.data = data
        self.hashing = hashing
        self.test_code = np.array([])
        self.nearest = []
        self.result = []

    def get_test_code(self):
        test = []
        for test_user in self.data.test_data:
            code = self.hashing.hashing_to_code(test_user)
            test.append(code)
        self.test_code = np.asarray(test)

    def NN_search(self, selectNearest, similar_user_number=10):
        for test_user in self.test_code:
            my_neighbour = selectNearest.select(test_user, self.hashing.train_code, similar_user_number)
            self.nearest.append(my_neighbour)

    def recommend(self, numberOfRec=10):
        '''
        Can be overrode to implement other ways to calculate recommendation result
        :param numberOfRec: Number of results
        :return: Result of recommendation
        '''
        for neighbours in self.nearest:
            result = np.zeros((1, self.data.test_data.shape[1]))
            for sim, user in neighbours:
                result += sim * self.data.train_data[user]
            self.result.append(result.argsort()[-numberOfRec:])

    def measure(self, measurement):
        pass