__author__ = 'soloconte'

import numpy as np


class Data(object):
    def __init__(self, path):
        self.path = path

    def read(self):
        '''
        Read data according to path and convert data into np.array
        Should be overrode for different data sources
        DO NOT call this function directly
        :return: Data read from path as np.array
        '''
        pass

