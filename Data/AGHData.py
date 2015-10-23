__author__ = 'soloconte'

from scipy import io
from Data import Data


class AGHData(Data):

    def __init__(self, path):
        super(AGHData, self).__init__(path)
        matPath = self.path + "mnist_split.mat"
        dataDict = io.loadmat(matPath)
        self.train_data = dataDict['traindata']
        self.test_data = dataDict['testdata']
        self.train_label = dataDict['traingnd'].T[0]
        self.test_label = dataDict['testgnd'].T[0]
        print "Data Prepared"

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_train_labels(self):
        return self.train_label

    def get_test_labels(self):
        return self.test_label


if __name__ == "__main__":
    mydata = AGHData("/home/soloconte/Codes/graph_hashing/Datasets/AGH/")
    print mydata.train_data.shape
    print mydata.test_data.shape
    print mydata.train_label.shape
    print mydata.test_label.shape