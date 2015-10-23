__author__ = 'soloconte'

import numpy as np
from Data import AGHData
from Hashing import AnchorGraphHashing
from Recommendation import AGHMnistRecommendation
from Experiments import Experiment


class AGHMnistExperiment(Experiment):
    def __init__(self, data_path):
        super(AGHMnistExperiment, self).__init__(data_path)
        self.predict = []
        self.ground_truth = np.zeros(1)

    def get_result(self):
        mydata = AGHData.AGHData(self.path)
        myhashing = AnchorGraphHashing.AnchorGraphHashing(mydata.get_train_data())
        myrecommendation = AGHMnistRecommendation.AGHMnistRecommendation(mydata, myhashing)
        myrecommendation.recommend()
        self.predict = np.asarray(myrecommendation.result)
        self.ground_truth = mydata.get_test_labels()

    def measure(self):
        n = self.ground_truth.shape[0]
        is_same = np.zeros(n, int)
        is_same[self.predict == self.ground_truth] = 1
        hit_num = is_same.sum()
        print "Total Test Sample : %d" % n
        print "Correct : %d" % hit_num
        print "Precision : %f" % (float(hit_num)/n)

    def conduct(self, times=10):
        for i in xrange(times):
            print "Experiment %d :" % (i+1)
            self.get_result()
            self.measure()


if __name__ == "__main__":
    path = "/home/soloconte/Codes/graph_hashing/Datasets/AGH/"
    experiment = AGHMnistExperiment(path)
    experiment.conduct(1)
