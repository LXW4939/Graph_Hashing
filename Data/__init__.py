__author__ = 'soloconte'

import numpy as np


class Data(object):
    def __init__(self, path):
        self.path = path
        self.data = np.array([])
        self.train_data = np.array([])
        self.test_data = np.array([])

    def read(self):
        '''
        Read data according to path and convert data into np.array
        Should be overrode for different data sources
        DO NOT call this function directly
        :return: Data read from path as np.array
        '''
        data_file = open(self.path)
        import pickle
        self.data = pickle.load(data_file)
        data_file.close()

    def split(self, prob=0.8):
        '''
        Split data into train data and test data
        DO NOT call this function directly
        :param prob: Probability of data to be split into train data
        :return: Train data and Test data as np.array
        '''
        train=[]
        test=[]
        n = self.data.shape[0]
        for i in xrange(n):
            train_prob = np.random.randint(0, 100)
            if train_prob >= 100 * prob:
               test.append(self.data[i])
            else:
                train.append(self.data[i])
        self.test_data = np.asarray(test)
        self.train_data = np.asarray(train)

    def get_split_data(self, prob=0.8):
        '''
        Just call this function to get data
        Read data and return split data
        :param prob: Probability of data split into train data
        :return: Train data and Test data as np.array
        '''
        self.read()
        self.split(prob)
        return self.train_data, self.test_data
