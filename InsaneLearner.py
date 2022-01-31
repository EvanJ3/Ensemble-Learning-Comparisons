import BagLearner as BL
import LinRegLearner as lrl
import numpy as np

class InsaneLearner:
    def __init__(self,verbose=False):
        self.verbose = verbose
        self.bags_list = [BL.BagLearner(learner=lrl.LinRegLearner,kwargs={},bags=20,boost=False,verbose=False) for i in range(0,20)]

    def add_evidence(self,x,y):
        [self.bags_list[i].add_evidence(x,y) for i in range(0,len(self.bags_list))]
            
    def query(self,x_test):
        preds_array = np.zeros((x_test.shape[0],20))
        for i in range(0,len(self.bags_list)):
            preds_array[:,i] = self.bags_list[i].query(x_test)
        return np.mean(preds_array,axis=1)