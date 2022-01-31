import RTLearner as rt
import DTLearner as dt
import LinRegLearner as lrl
import numpy as np

class BagLearner:
    
    def __init__(self,learner,kwargs,bags,boost=False,verbose=False):
        self.learner = learner
        self.bags = bags
        self.kwargs = kwargs
        self.boost = boost
        self.verbose = verbose
        self.learner_lister = self.initialize_learners()
        self.n_samples = None
        self.x_train = None
        self.y_train = None
    
    def initialize_learners(self):
        learner_list = []
        kwargs = self.kwargs
        for i in range(0,self.bags):
            learner_list.append(self.learner(**kwargs))

        
        return learner_list
    
    def add_evidence(self,x,y):
        self.n_samples = y.shape[0]
        self.x_train = x
        self.y_train = y
        data_indexes = np.random.choice(self.n_samples,self.n_samples*self.bags,replace=True).reshape(self.n_samples,self.bags)
        for i in range(0,len(self.learner_lister)):
            self.learner_lister[i].add_evidence(self.x_train[data_indexes[:,i],:],self.y_train[data_indexes[:,i]])
    
    def query(self,x_test):
        preds_array = np.zeros((x_test.shape[0],len(self.learner_lister)))
        for i in range(0,len(self.learner_lister)):
            preds_array[:,i] = self.learner_lister[i].query(x_test)
        return np.mean(preds_array,axis=1)