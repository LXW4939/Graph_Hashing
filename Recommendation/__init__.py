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
        self.train_code = self.hashing.hashing_to_code(self.data.train_data)
        self.test_code = self.hashing.hashing_to_code(self.data.test_data)
        print "Code Generated"
        # #######################  temp code  #######################################
        # import pickle
        # train_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/train_code", "w")
        # pickle.dump(self.train_code, train_file)
        # train_file.close()
        # test_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/test_code", "w")
        # pickle.dump(self.test_code, test_file)
        # test_file.close()
        # #########################################################################3

    def neighbour_search(self, select_nearest, similar_user_number=10):
        for test_user in self.test_code:
            my_neighbour = select_nearest(test_user, self.train_code, similar_user_number)
            self.nearest.append(my_neighbour)
        print "Neighbours Found"

    def recommend(self, select_nearest=SelectNearest.select_radius, similar_user_number=10, number_of_rec=10):
        self.generate_code()
        self.neighbour_search(select_nearest)

        # #############################  temp code  #############################
        # import pickle
        # data_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/nearest", "w")
        # pickle.dump(self.nearest, data_file)
        # data_file.close()
        # #######################################################################

        # #############################  temp code  #############################
        # import pickle
        # data_file = open("/home/soloconte/Codes/graph_hashing/Datasets/AGH/nearest")
        # self.nearest = pickle.load(data_file)
        # data_file.close()
        # #######################################################################
