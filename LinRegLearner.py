import numpy as np 

class LinRegLearner(object):
    def __init__(self, verbose=False):
        pass
    
    def add_evidence(self, data_x, data_y):
        new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data_x[:,0:data_x.shape[1]] = data_x
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(new_data_x, data_y, rcond=None)
        
    def query(self, points):
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]