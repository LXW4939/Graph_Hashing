__author__ = 'soloconte'

import numpy as np
from Data import AGHData
from Hashing import AnchorGraphHashing
from Recommendation import AGHMnistRecommendation
from Experiments import Experiment

class AGH_MNIST_Experiment(Experiment):


    def __init__(self, path):
        super(AGH_MNIST_Experiment, self).__init__(path)
        self.predict = []
        self.ground_truth = []

    def get_result(self):
        myData = AGHData.AGHData(self.path)
        myHashing = AnchorGraphHashing.AnchorGraphHashing(myData.get_train_data())
        myRecommendation = AGHMnistRecommendation.AGHMnistRecommendation(myData, myHashing)
        myRecommendation.recommend()
        self.predict = np.asarray(myRecommendation.result)
        self.ground_truth = myData.get_test_labels()

    def measure(self):
        n = self.ground_truth.shape[0]
        isSame = np.zeros(n, int)
        isSame[self.predict==self.ground_truth] = 1
        hitNum = isSame.sum()
        print "Total Test Sample : %d" % n
        print "Correct : %d" % hitNum
        print "Precision : %f" % (float(hitNum)/n)

    def conduct(self, times=10):
        for i in xrange(times):
            print "Experiment %d :" % (i+1)
            self.get_result()
            self.measure()


if __name__ == "__main__":
    path = "/home/soloconte/Codes/graph_hashing/Datasets/AGH/"
    experiment = AGH_MNIST_Experiment(path)
    experiment.conduct()