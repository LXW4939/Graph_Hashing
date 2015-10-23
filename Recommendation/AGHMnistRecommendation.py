__author__ = 'soloconte'

from Recommendation import Recommendation
from Utility import SelectNearest
import numpy as np


class AGHMnistRecommendation(Recommendation):
    def recommend(self, select_nearest=SelectNearest.select_radius, similar_user_number=10, number_of_rec=10):
        super(AGHMnistRecommendation, self).recommend(select_nearest, similar_user_number, number_of_rec)
        self.result = []
        for i in xrange(self.data.get_test_labels().shape[0]):
            score = np.zeros(10)
            for sim, user in self.nearest[i]:
                score[self.data.get_train_labels()[user]-1] += sim
            self.result.append(score.argmax()+1)
        print "Recommendation Finished"
