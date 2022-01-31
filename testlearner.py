import numpy as np
import BagLearner as bag
import DTLearner as dt
import RTLearner as rt
import matplotlib.pyplot as plt
import time
import sys


class Experiment():
    def __init__(self,data):
        self.data = data
        self.train_portion = .6
        
        self.num_leafs = 50
        self.DT_RMSE_IN_Out = None
        self.DT_RMSE_Diff = None
        self.DT_L1_IN_Out = None
        self.DT_L1_Diff = None
        
        self.RT_RMSE_IN_Out = None
        self.RT_RMSE_Diff = None
        self.RT_L1_IN_Out = None
        self.RT_L1_Diff = None
        
        self.num_loops = 10
        self.num_timed_loops = 10
        self.timed_leaves = range(1,self.num_leafs+1,2)
        self.x_train_fake = np.random.randn(10000,8)
        self.rt_train_times = None
        self.dt_train_times = None
        self.dt_pred_times = None
        self.rt_pred_times = None
        self.bag_size = 20
        self.timed_x_train,self.timed_y_train,self.timed_x_test,self.timed_y_test = self.train_test_split()
        
        self.BAG_RMSE_IN_Out = None
        self.BAG_RMSE_Diff = None

    def train_test_split(self):
        num_train_rows = int(self.train_portion * self.data.shape[0])
        num_test_rows = self.data.shape[0] - num_train_rows
        random_indexes = np.random.choice(self.data.shape[0],size=self.data.shape[0],replace=False)
        train_indexes = random_indexes[:num_train_rows]
        test_indexes = random_indexes[num_train_rows:num_test_rows+num_train_rows]
        x_train = self.data[train_indexes, 0:-1]
        y_train = self.data[train_indexes, -1]
        x_test = self.data[test_indexes, 0:-1]
        y_test = self.data[test_indexes, -1]
        if self.train_portion == 1:
            x_test = x_train
            y_test = y_train
        return x_train,y_train,x_test,y_test
    
    def train_DT_RT(self):
        bag_RMSE_array = np.zeros((self.num_loops,2,self.num_leafs))
        dt_RMSE_array = np.zeros((self.num_loops,2,self.num_leafs))
        dt_L1_array = np.zeros((self.num_loops,2,self.num_leafs))
        rt_L1_array = np.zeros((self.num_loops,2,self.num_leafs))
        for j in range(0,self.num_loops):
            x_train,y_train,x_test,y_test = self.train_test_split()
            for i in range(1,self.num_leafs+1):
                
                dt_learner = dt.DTLearner(leaf_size=i,verbose=False)
                rt_learner = rt.RTLearner(leaf_size=i,verbose=False)
                bag_learner = bag.BagLearner(dt.DTLearner,{'leaf_size':i},bags=self.bag_size)
                
                dt_learner.add_evidence(x_train,y_train)
                rt_learner.add_evidence(x_train,y_train)
                bag_learner.add_evidence(x_train,y_train)
                
                dt_train_preds = dt_learner.query(x_train)
                rt_train_preds = rt_learner.query(x_train)
                bag_train_preds = bag_learner.query(x_train)
                
                dt_test_preds = dt_learner.query(x_test)
                rt_test_preds = rt_learner.query(x_test)
                bag_test_preds = bag_learner.query(x_test)

                dt_RMSE_array[j,0,i-1] = np.sqrt(np.mean((dt_train_preds - y_train)**2.0))
                dt_RMSE_array[j,1,i-1] = np.sqrt(np.mean((dt_test_preds - y_test)**2.0))
                

                dt_L1_array[j,0,i-1] = np.sum(np.abs(dt_train_preds - y_train))
                dt_L1_array[j,1,i-1] = np.sum(np.abs(dt_test_preds - y_test))
                
                rt_L1_array[j,0,i-1] = np.sum(np.abs(rt_train_preds - y_train))
                rt_L1_array[j,1,i-1] = np.sum(np.abs(rt_test_preds - y_test))
                
                bag_RMSE_array[j,0,i-1] = np.sqrt(np.mean((bag_train_preds - y_train)**2.0))
                bag_RMSE_array[j,1,i-1] = np.sqrt(np.mean((bag_test_preds - y_test)**2.0))
                


            self.DT_RMSE_IN_Out = np.mean(dt_RMSE_array,axis=0)
            self.DT_RMSE_Diff = self.DT_RMSE_IN_Out[1,:] - self.DT_RMSE_IN_Out[0,:]
            
            self.DT_L1_IN_Out = np.mean(dt_L1_array,axis=0)
            self.DT_L1_Diff = self.DT_L1_IN_Out[1,:] - self.DT_L1_IN_Out[0,:]
            
            self.RT_L1_IN_Out = np.mean(rt_L1_array,axis=0)
            self.RT_L1_Diff = self.RT_L1_IN_Out[1,:] - self.RT_L1_IN_Out[0,:]
            
            self.BAG_RMSE_IN_Out = np.mean(bag_RMSE_array,axis=0)
            self.BAG_RMSE_Diff = self.BAG_RMSE_IN_Out[1,:] - self.BAG_RMSE_IN_Out[0,:]
    
    
    def Time_DT_Train(self):
        dt_train_times = np.zeros((len(self.timed_leaves),))
        dt_pred_times = np.zeros((len(self.timed_leaves),))
        for j in range(0,len(self.timed_leaves)):
            tot_train_time = 0.0
            tot_pred_time = 0.0
            for i in range(0,self.num_timed_loops):
                dt_time_start = time.perf_counter() 
                dt_learner = dt.DTLearner(leaf_size=self.timed_leaves[j])
                dt_learner.add_evidence(self.timed_x_train,self.timed_y_train)
                dt_time_end = time.perf_counter() 
                dt_pred_start = time.perf_counter() 
                dt_preds = dt_learner.query(self.x_train_fake)
                dt_pred_end = time.perf_counter() 
                dt_learner = None
                dt_preds = None
                dt_train_time = dt_time_end - dt_time_start
                dt_pred_time = dt_pred_end - dt_pred_start
                tot_train_time+= dt_train_time
                tot_pred_time+= dt_pred_time
            avg_leaf_train_time = tot_train_time/self.num_timed_loops
            avg_leaf_pred_time = tot_pred_time/self.num_timed_loops
            dt_train_times[j] = avg_leaf_train_time
            dt_pred_times[j] = avg_leaf_pred_time
        self.dt_train_times = dt_train_times
        self.dt_pred_times = dt_pred_times
    
    def Time_RT_Train(self):
        rt_train_times = np.zeros((len(self.timed_leaves),))
        rt_pred_times = np.zeros((len(self.timed_leaves),))
        for j in range(0,len(self.timed_leaves)):
            tot_train_time = 0.0
            tot_pred_time = 0.0
            for i in range(0,self.num_timed_loops):
                rt_time_start = time.perf_counter() 
                rt_learner = rt.RTLearner(leaf_size=self.timed_leaves[j])
                rt_learner.add_evidence(self.timed_x_train,self.timed_y_train)
                rt_time_end = time.perf_counter() 
                rt_pred_start = time.perf_counter() 
                rt_preds = rt_learner.query(self.x_train_fake)
                rt_pred_end = time.perf_counter() 
                rt_learner = None
                rt_preds = None
                rt_train_time = rt_time_end - rt_time_start
                rt_pred_time = rt_pred_end - rt_pred_start
                tot_train_time+= rt_train_time
                tot_pred_time+= rt_pred_time
            avg_leaf_train_time = tot_train_time/self.num_timed_loops
            avg_leaf_pred_time = tot_pred_time/self.num_timed_loops
            rt_train_times[j] = avg_leaf_train_time
            rt_pred_times[j] = avg_leaf_pred_time
        self.rt_train_times = rt_train_times
        self.rt_pred_times = rt_pred_times


    def Gen_Plot_1(self,save=True):
        idx = np.arange(1,self.num_leafs+1)
        plt.figure(figsize=(15,7))

        ax1 = plt.subplot(121)
        ax1.plot(idx,self.DT_RMSE_IN_Out[0,:],label='In-Sample Error',color='blue')
        ax1.plot(idx,self.DT_RMSE_IN_Out[1,:],label='Out-Sample Error',color='blue',linestyle='dashed')
        ax1.title.set_text('Decision Tree Out/In Sample RMSE by Leaf Size')
        ax1.legend()
        ax1.set_ylabel('Average Residual Mean Squared Error')
        ax1.set_xlabel('Leaf Size')

        ax2 = plt.subplot(122)
        ax2.plot(idx,self.DT_RMSE_Diff,label='Average RMSE Differences',color='blue')
        ax2.legend()
        ax2.title.set_text('Decision Tree Out/In Sample RMSE Differences by Leaf Size')
        ax2.set_xlabel('Leaf Size')
        ax2.set_ylabel('Average Out & In Sample RMSE Difference')
        if save:
            plt.savefig('Figure_1_DT_RMSE_Error')
        else:
            plt.show()

    def Gen_Plot_2(self,save=True):
        plt.figure(figsize=(15,7))
        idx = np.arange(1,self.num_leafs+1)
        ax1 = plt.subplot(121)
        ax1.plot(idx,self.BAG_RMSE_IN_Out[0,:],label='Bag Learner In-Sample Error',color='orange')
        ax1.plot(idx,self.BAG_RMSE_IN_Out[1,:],label='Bag Learner Out-Sample Error',linestyle='dashed',color='orange')
        ax1.plot(idx,self.DT_RMSE_IN_Out[0,:],label='DT Learner In Sample Error',color='blue')
        ax1.plot(idx,self.DT_RMSE_IN_Out[1,:],label='DT Learner Out Sample Error',color='blue',linestyle='dashed')
        ax1.legend()
        ax1.title.set_text('Bagged Decison Trees Out/In Sample RMSE')
        ax1.set_ylabel('Average Residual Mean Squared Error')
        ax1.set_xlabel('Leaf Size')
        
        ax2 = plt.subplot(122)
        ax2.plot(idx,self.BAG_RMSE_Diff,label='20 Bag Learner',color='orange')
        ax2.plot(idx,self.DT_RMSE_Diff,label='DT Leaner', color='blue')
        ax2.legend()
        ax2.title.set_text('Bagged Decison Trees Out/In Sample RMSE Difference')
        ax2.set_ylabel('Average Out & In Sample RMSE Difference')
        ax2.set_xlabel('Leaf Size')
        if save:
            plt.savefig('Figure_2_Bagged_RMSE_Error')
        else:
            plt.show()
        
       
    def Gen_Plot_3(self,save=True):
        idx = np.arange(1,self.num_leafs+1)
        plt.figure(figsize=(15,7))

        ax1 = plt.subplot(121)
        ax1.plot(idx,self.DT_L1_IN_Out[0,:],label='Decision Tree In-Sample Error',color='blue')
        ax1.plot(idx,self.DT_L1_IN_Out[1,:],label='Decision Tree Out-Sample Error',color='blue',linestyle='dashed')
        ax1.plot(idx,self.RT_L1_IN_Out[0,:],label='Random Tree In-Sample Error',color='red')
        ax1.plot(idx,self.RT_L1_IN_Out[1,:],label='Random Tree Out-Sample Error',color='red',linestyle='dashed')
        ax1.title.set_text('Decision vs. Random Trees Out & In Sample L1 Error')
        ax1.legend()
        ax1.set_ylabel('Average Sum of Absolute Error')
        ax1.set_xlabel('Leaf Size')
        
        ax2 = plt.subplot(122)
        ax2.plot(idx,self.DT_L1_Diff,label='Decison Tree',color='blue')
        ax2.plot(idx,self.RT_L1_Diff,label='Random Tree',color='red')
        ax2.legend()
        ax2.title.set_text('Decision vs. Random Trees Out & In Sample L1 Differences')
        ax2.set_xlabel('Leaf Size')
        ax2.set_ylabel('Average Out & In Sample Absolute Error Difference')
        if save:
            plt.savefig('Figure_3_DT_vs_RT_L1_Error')
        else:
            plt.show()
    
    
    def Gen_Plot_4(self,save=True):
        plt.figure(figsize=(15,7))
        ax1 = plt.subplot(121)
        ax1.plot(self.timed_leaves,self.dt_train_times,label='Decision Tree Training Time',color='blue')
        ax1.plot(self.timed_leaves,self.rt_train_times,label='Random Tree Training Time',color='red')
        ax1.title.set_text('Decison vs. Random Tree Training Time')
        ax1.set_ylabel('Average Training Time in Seconds')
        ax1.set_xlabel('Leaf Size')
        ax1.legend()
        
        ax2 = plt.subplot(122)
        ax2.plot(self.timed_leaves,self.dt_pred_times,label='Decision Tree Inference Time',color='blue')
        ax2.plot(self.timed_leaves,self.rt_pred_times,label='Random Tree Infernce Time', color='red')
        ax2.title.set_text('Decision vs. Random Tree Inference Time')
        ax2.set_ylabel('Average Inference Time in Seconds')
        ax2.set_xlabel('Leaf Size')
        ax2.legend()
        
        if save:
            plt.savefig('Figure_4_DT_vs_RT_Clock_Time')
        else:
            plt.show()
        
    def Train(self):
        self.train_DT_RT()
        time.sleep(1)
        self.Time_DT_Train()
        time.sleep(1)
        self.Time_RT_Train()
        
        
    def Plot_All(self,save=True):
        if save:
            self.Gen_Plot_1(save=True)
            self.Gen_Plot_2(save=True)
            self.Gen_Plot_3(save=True)
            self.Gen_Plot_4(save=True)
        else:
            self.Gen_Plot_1(save=False)
            self.Gen_Plot_2(save=False)
            self.Gen_Plot_3(save=False)
            self.Gen_Plot_4(save=False)
            
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    if len(sys.argv) != 2:  		  	   		   	 			  		 			 	 	 		 		 	
        print("Usage: python testlearner.py <filename>")  		  	   		   	 			  		 			 	 	 		 		 	
        sys.exit(1)  		  	   		   	 			  		 			 	 	 		 		 	
    inf = open(sys.argv[1])
    #data = np.genfromtxt('Istanbul.csv',dtype=float,delimiter=',',skip_header=1,usecols=(1,2,3,4,5,6,7,8,9))
    data = np.genfromtxt(inf,dtype=float,delimiter=',',skip_header=1,usecols=(1,2,3,4,5,6,7,8,9))
    Experiment_Object = Experiment(data)
    Experiment_Object.Train()
    Experiment_Object.Plot_All()