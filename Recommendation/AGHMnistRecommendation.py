__author__ = 'soloconte'

from Recommendation import Recommendation
from Utility import SelectNearest
import numpy as np


class AGHMnistRecommendation(Recommendation):
    def recommend(self, selectNearest=SelectNearest.selectRadius, similar_user_number=10, numberOfRec=10):
        super(AGHMnistRecommendation, self).recommend(selectNearest, similar_user_number, numberOfRec)
        self.result = []
        for i in xrange(len(self.data.get_test_labels().shape[0])):
            score = np.zeros(10)
            for sim, user in self.nearest[i]:
                score[self.data.get_train_labels()[user]-1] += sim
            self.result.append(score.argmax()+1)
        print "Reccomendation Finished"


