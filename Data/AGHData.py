__author__ = 'soloconte'

from scipy import io
from Data import Data


class AGHData(Data):

    def __init__(self, path):
        super(AGHData, self).__init__(path)
        mat_path = self.path + "mnist_split.mat"
        data_dict = io.loadmat(mat_path)
        self.train_data = data_dict['traindata']
        self.test_data = data_dict['testdata']
        self.train_label = data_dict['traingnd'].T[0]
        self.test_label = data_dict['testgnd'].T[0]
        print "Data Prepared"

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_train_labels(self):
        return self.train_label

    def get_test_labels(self):
        return self.test_label
