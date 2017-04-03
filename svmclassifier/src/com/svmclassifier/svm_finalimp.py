import numpy as np
from com.svmclassifier.AD_Machine import AD_Machine
import pandas as pd
from collections import OrderedDict
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from pickle import dump
from pickle import load
import warnings
warnings.filterwarnings("ignore")
class AD_Machine_SVM(AD_Machine):
    
    # constructor that initializes the parameters
    def __init__(self ):
        self.threshold = 4
        self.block_size = 100
        self.sub_block_size = 50
        self.num_sub_blocks = 6
        self.nu = 0.15
        self.kernel = "rbf"
        self.gamma = 0.0001
        self.idsize = 3378
        self.model_file = None
        self.train_called_flag = 0
        self.load_model_called_flag = 0
    
    ## custom scorer created for developing cross validation strategy on reducing the false alarm rate on training dataset
    ## input : predicted output and actual output
    ## returns the false alarm rate or 0 if none
    def scorer(self,predict_output_df,actual_df):
        try:
            tn, fp, fn, tp = confusion_matrix(actual_df, predict_output_df).ravel()
            false_alarm = fp / (fp+ tp)
            return false_alarm
        except ValueError:
            return 0
    
    ## Function to build the model along with grid search for tuning hyper parameters with cross validation on false alarm rate
    # input : final data structure
    # output : returns SVM model
    def create_model_svm(self,final_data_structure):
        all_data_frames = []
        #print(final_data_structure[0])
        for block_id , sub_block_list in final_data_structure.items():
            for df in sub_block_list:
                #print(df)
                all_data_frames.append(df)
        #*************** code for tuning the SVM using grid search***********
        #actual_df = [0]*len(all_data_frames)
        #loss  = make_scorer(self.scorer, greater_is_better=False)      
        #clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma )
        #parameters = {'nu': np.arange(0.0001,0.01, 0.0002)}
        #print("building the model")
        #grid = GridSearchCV(clf, parameters, cv = 4, scoring = loss)
        #model_svm2 = grid.fit(all_data_frames , actual_df)
        #print("completed building the model")
        #model_svm3 = model_svm2.best_estimator_
        #print(model_svm2.best_params_)
        clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma )
        model_svm = clf.fit(all_data_frames)
        print("training is completed")
        self.model_file = model_svm
        return model_svm      
         
    # input : model and the data structure
    # output : Prints the model metrics on console  
    def model_metrics(self, type_run, model_svm, final_data_structure):
        predict_output = []
        total_bad_blocks = 0
        total_blocks_passed = len(final_data_structure)
        for id_no, block_list in final_data_structure.items():
            temp = [int(model_svm.predict(x)) for x in block_list]
            if(temp.count(-1) > self.threshold):
                print("bad block present at",(id_no,id_no+100))
                total_bad_blocks += 1
                predict_output.append(1)
            else:
                predict_output.append(0)                 
        if (type_run == 'training'):
            print("total blocks passed",total_blocks_passed," with abnormal block count",total_bad_blocks)
            print("training abnormality rate = ", total_bad_blocks/total_blocks_passed)
            predict_output_df = pd.DataFrame(predict_output)
            predict_output_df.to_csv('training_output.csv',index = False, header= False)
        else:
            print("total blocks passed",total_blocks_passed,"with abnormal block count",total_bad_blocks)
            print("testing abnormality rate = ", total_bad_blocks/total_blocks_passed)
            return predict_output
                
    # input : user data stream 
    # output: creates a dictionary of final data structure {block id:[sub-block list of size 50]...}    
    def create_nparray(self,data_stream):
        final_data_structure = OrderedDict()
        temp_idlist = []
        idarr = [0]*(self.idsize)
        for id_no in data_stream:
            try:
                temp_idlist.append(int(id_no.strip()))
            except ValueError:
                pass
        data_stream.close()
        for i in range(0,len(temp_idlist),self.block_size):
            temp_arr = temp_idlist[i:i+100]
            temp_subarray= []
            for j in range(0,self.num_sub_blocks):
                temp_sb = temp_arr[j*10: j*10 +self.sub_block_size]
                for k in temp_sb:
                    idarr[k-1] += 1
                temp_subarray.append(idarr)
                idarr = [0]*self.idsize
            final_data_structure[i]= temp_subarray
            temp_subarray = []                  
        return final_data_structure
    
    #input:Implements the abstract class for loading model file 
    #output : sets the model object to the class
    def load_model(self, model_file):
        self.model_file = load(open(model_file, 'rb'))
        self.load_model_called_flag = 1
    
    #input:Implements abstract class for training
    #output: prints out the metrics of the training and returns the control with 0 to driver function
    def train(self, data_stream, delimiter=',', sample_size=1, save_model_file=None):
        self.block_size = sample_size 
        final_data_structure = self.create_nparray(data_stream)
        model_svm = self.create_model_svm(final_data_structure) 
        dump(model_svm,open(save_model_file, 'wb'))   
        self.model_metrics('training',model_svm, final_data_structure)
        self.train_called_flag = 1
        return 0
    
    #input:Implements the abstract class function for testing
    # output: prints out the metrics with bad block details and writes the predicted results to output stream
    def predict(self, data_stream, predict_stream, delimter=',', sample_size=1, mode='batch'):
        if(not self.train_called_flag and not self.load_model_called_flag):
            if(not self.train_called_flag):
                raise Exception("predict called before training")
            elif(not self.load_model_called_flag):
                raise Exception("predict called before loading the model")
            else:
                raise Exception("predict called before both loading and testing the model")
        testing_data_structure = self.create_nparray(data_stream)
        predict_output = self.model_metrics('testing', self.model_file, testing_data_structure)
        for item in predict_output:
            predict_stream.write(str(item))
        return 0
