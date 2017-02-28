import csv
import pandas as pd
from collections import OrderedDict
import numpy as np
import io
import glob
import os
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from pickle import dump
from pickle import load
from sqlalchemy.sql.expression import false

class Training():
    
    # constructor that initializes the parameters
    def __init__(self ):
        self.PATH = r'C:\Users\bharadwaj\Desktop\masquerade-data'
        self.threshold = 4
        self.block_size = 100
        self.sub_block_size = 50
        self.num_sub_blocks = 6
        self.nu = 0.001
        self.kernel = "rbf"
        self.gamma = 0.0001
        self.database = self.create_database()
    
    # creates a database of all the user files and used to dump in future. 
    def create_database(self):
        database = {}
        word_id = 0
        files = [file for file in glob.glob(self.PATH + '/**/*', recursive=True)]
        for file_name in files:
            with io.open(file_name, 'r') as file_lines:
                for line in file_lines:
                    word = line.strip().lower()
                    if (word not in database.keys()):
                        database[word] = word_id
                        word_id += 1
                    else:
                        continue
        dump(database, open('database.dict','wb'))
        with open('dictionary.csv',  'w', newline='') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f)
            w.writerows(database.items())
        return database 
    
    # input : entire file as dataframe
    # output: blocks of size 100 and sub-blocks of size 50.
    def create_blocks(self,data_frame): 
        dict_dataframes = {}
        len_df = len(data_frame.axes[0])
        for i in range(0,len_df, self.block_size):
            sub_blocks = []
            temp_df = data_frame[i:i+self.block_size]
            for j in range(0, self.num_sub_blocks):
                sub_blocks.append(temp_df[j*10:j*10+self.sub_block_size])
            dict_dataframes[i] = sub_blocks
        return dict_dataframes
    
    # input : each sublist of size 50
    # output : a trainable sparse data frame of size length of database with columns 'id' and count. 
    # if 'id' is not there in the 10 window span, it has count 0 , else the count is populated by window aggregation.        
    def create_training_dataframe(self,each_sublist):
        len_df = len(each_sublist.axes[0])
        word_count_dict = dict(self.database)
        temp_df1 = each_sublist.groupby('id').sum()
        #print(temp_df1)
        id_columns = list(word_count_dict.values())
        id_columns = sorted(id_columns)
        training_df = pd.DataFrame(columns= id_columns , index = [1])
        training_df.ix[1] = 0
        for ir in temp_df1.itertuples():
            training_df.ix[1][ir[0]] = ir[1]
        return training_df

    # input : complete dataframe from a training/ testing dataset for modelling.
    # output: will call SVM model from here.        
    def create_data_format(self,data_frame):
        final_data_structure = {}  # {block_number : [{sub-block_number : df of size 50}]} 
        dict_blocks = self.create_blocks(data_frame) # creating a dict of data blocks { block_id : sub_blocks of 50 size}
        for block_number in dict_blocks.keys():
            sub_lists = dict_blocks[block_number]
            sublist_dfs=[]
            for each_sublist in sub_lists:
                training_df = self.create_training_dataframe(each_sublist)
                sublist_dfs.append(training_df)
            final_data_structure[block_number] = sublist_dfs
        return final_data_structure
    
    # input : final data structure
    # output : model of SVM with tuning code included using grid search cross validating the accuracy
    def create_model_svm(self,final_data_structure):
        id_columns = list(self.database.values())
        id_columns = sorted(id_columns)
        all_data_frames = pd.DataFrame(columns= id_columns)
        for block_id , sub_block_list in final_data_structure.items():
            #print(sub_block_list)
            for df in sub_block_list:
                all_data_frames = all_data_frames.append(df, ignore_index = True)
        
        #*************** code for tuning the SVM using grid search***********        
        #clf = svm.OneClassSVM()
        #parameters = {'kernel': ['linear', 'rbf'], 'nu': np.arange(0.0001,0.01, 0.0002),'gamma': np.arange(0.0001,0.01, 0.0002)}
        #print(model_svm)
        #grid = GridSearchCV(clf, parameters, scoring = 'accuracy')
        #model_svm2 = grid.fit(all_data_frames , pd.DataFrame([1]*len(all_data_frames)))
        #self.kernel = model_svm2.best_params_['kernel']
        #self.gamma = model_svm2.best_params_['gamma']
        #self.nu = model_svm2.best_params_['nu']
        clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma )
        model_svm = clf.fit(all_data_frames)
        return model_svm        
    
    def calculate_testing_metrics(self,predict_output_df , user_no):
        predict_output_df.to_csv('predicted_output.csv')
        outputdf = pd.read_csv('actual_output.csv')
        actual_df = outputdf.ix[:,user_no-1]
        actual_df.to_csv('actual-output.csv')
        tn, fp, fn, tp = confusion_matrix(actual_df, predict_output_df).ravel()
        print(tn,fp,fn,tp)
        accuracy_rate = (tn + tp)/(tn+tp+fn+fp)
        misses = (fn+fp)/(tn+tp+fn+fp)
        false_alarm = fp / (fp+ tp)
        print("accuracy rate is", accuracy_rate)
        print("misses rate is ", misses)
        print("false alarm rate is", false_alarm)
         
    # input : final metrics strucutre 
    # output: prints the block id, sub block id and the indexes at which bad blocks are found 
    def calculate_statistics(self,metrics_structure):
        metrics_structure = dict(OrderedDict(sorted(metrics_structure.items())))
        predict_output_df =  pd.DataFrame([0]*100)
        total_bad_blocks = 0
        total_blocks_passed = len(metrics_structure)*self.num_sub_blocks
        for block_id , result_list in metrics_structure.items():
            num_bad_rows = result_list.count(-1)
            if(num_bad_rows > self.threshold):
                total_bad_blocks += 1
                predict_output_df.ix[block_id/100 ] = 1
                print("bad block of rows",(block_id, block_id+100))
        print("total number of blocks passed",total_blocks_passed, "total number of bad blocks", total_bad_blocks)
        misclassificaiton_rate = total_bad_blocks / total_blocks_passed
        print("abnormality rate is ", misclassificaiton_rate*100)                
        #print(predict_output_df.ix[60:63] )
        return predict_output_df
    
    # input : model and the data structure
    # output : Prints the model metrics on console  
    def model_metrics(self, x , model_svm, final_data_structure):
        metrics_structure = {}
        for block_id , df_list in final_data_structure.items():
                fifty_array = []
                for block in df_list:
                    with open(os.path.join(x + '_df' + '.csv'), 'a') as f:
                        block.to_csv(f, header=False)
                        fifty_array.append(int(model_svm.predict(block)))
                metrics_structure[block_id] = fifty_array  
        predict_output_df = self.calculate_statistics(metrics_structure)              
        return dict(metrics_structure) , predict_output_df    
                
    # input : user file 
    # output : dataframe of wordlists and counts with 1    
    def create_nparray(self,file_name):
        word_list = []
        with io.open(file_name, 'r') as file_lines1:
            for line in file_lines1:
                word = line.strip().lower()
                word_id = self.database[word]
                word_list.append((word_id, 1))
        labels = ['id' , 'count']
        data_frame = pd.DataFrame.from_records(word_list, columns = labels) 
        final_data_structure = self.create_data_format(data_frame)        
        return final_data_structure
    
    #input : driver program for training which takes user name as input
    # output : prints out the metrics of the training with bad blocks details
    def training_model(self, training_user, user_no):
        training_file = os.path.join(self.PATH,training_user) 
        final_data_structure = self.create_nparray(training_file)
        model_svm = self.create_model_svm(final_data_structure)  
        dump(model_svm,open('model.svm', 'wb'))   
        metrics_structure ,predict_output_df = self.model_metrics('training',model_svm, final_data_structure)
    #input : driver program for testing which takes user name as input
    # output : prints out the metrics with bad block details
    
    def model_testing(self, testing_user, user_no):
        load(open('database.dict','rb'))
        testing_file = os.path.join(self.PATH, testing_user)
        testing_data_structure = self.create_nparray(testing_file)
        trained_model = load(open('model.svm', 'rb'))
        testing_metrics_structure ,predict_output_df = self.model_metrics('testing', trained_model, testing_data_structure)
        self.calculate_testing_metrics( predict_output_df, user_no)
        

# creating object of training class and calling functions
trn =  Training()
#trn.training_model('user43_train', 43)
trn.model_testing('user43_test', 43)
