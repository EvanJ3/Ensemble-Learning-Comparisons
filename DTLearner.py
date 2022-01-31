import numpy as np

class DTLearner:
    
    def __init__(self,leaf_size=1,verbose=False): 
        self.x_data = None
        self.y_labels = None
        self.feature_criterion = self.absolute_correlation
        self.split_criterion = self.median_split
        self.node_array = None
        self.max_leaves = leaf_size
        self.verbose = verbose

    def build_tree(self,x,y):

        if y.shape[0] <= self.max_leaves:
            return np.array([[None,np.mean(y),None,None]])
        if np.unique(y).size == 1:
            return np.array([[None,np.mean(y),None,None]])
        else:
            split_index = self.feature_criterion(x,y)
            split_val = self.split_criterion(x,split_index)
            left_tree_x = x[x[:,split_index]<=split_val]
            left_tree_y = y[x[:,split_index]<=split_val]
            right_tree_x = x[x[:,split_index]>split_val]
            right_tree_y = y[x[:,split_index]>split_val]
            if left_tree_y.size == 0 or right_tree_y.size==0:
                root = np.array([[None,np.mean(y),None,None]])
                return root
            else:
                left_tree = self.build_tree(left_tree_x,left_tree_y)
                right_tree = self.build_tree(right_tree_x,right_tree_y)
                root = np.array([[split_index,split_val,1,left_tree.shape[0]+1]])
                return np.append(root,np.append(left_tree,right_tree,axis=0),axis=0)
    
    def add_evidence(self,data_x,data_y):
        self.x_data = data_x
        self.y_labels = data_y
        self.node_array =self.build_tree(self.x_data,self.y_labels)
    
    def median_split(self,x,split_index):
        return np.median(x[:,split_index])
    
    def absolute_correlation(self,x,y):
        abs_corr_coefs = np.abs(np.corrcoef(x,y=y,rowvar=False)[:-1,-1])
        max_corr = np.nanargmax(abs_corr_coefs)
        return max_corr
    
    def query(self,x_test):
        preds_array = np.zeros((x_test.shape[0]))
        pred_index = 0
        for rows in x_test:
            current_index = 0
            row_solved = False
            while not(row_solved):
                feature_split_index = self.node_array[current_index,0]
                split_val = self.node_array[current_index,1]
                if feature_split_index is None:
                    preds_array[pred_index] = split_val
                    pred_index+=1
                    row_solved = True

                elif rows[int(feature_split_index)] <= split_val:
                    current_index += 1
                else:
                    current_index += int(self.node_array[current_index,3])
        return preds_array