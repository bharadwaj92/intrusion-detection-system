import csv
import pandas as pd
import numpy as np
from collections import OrderedDict
import io
import glob
import os
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from pickle import dump
from pickle import load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

class Training():
    
    # constructor that initializes the parameters
    def __init__(self , user_no):
        self.PATH = r'C:\Users\bharadwaj\Desktop\masquerade-data'
        self.threshold = 4
        self.block_size = 100
        self.sub_block_size = 50
        self.num_sub_blocks = 6
        self.nu = 0.1
        self.kernel = "rbf"
        self.gamma = 0.0001
        self.user_no = user_no
        self.testing_actual_values = ""
        self.training_file= ""
        self.testing_file = ""
        self.database = self.create_database()
        self.testing_actual_values, self.training_file, self.testing_file = self.create_test_train(user_no)
        self.n_components = 50
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
        dict_dataframes = OrderedDict()
        len_df = len(data_frame.axes[0])
        for i in range(0,len_df, self.block_size):
            sub_blocks = []
            temp_df = data_frame[i:i+self.block_size]
            for j in range(0, self.num_sub_blocks):
                sub_blocks.append(temp_df[j*10:j*10+self.sub_block_size])
            dict_dataframes[i] = sub_blocks
        #print(dict_dataframes)
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
        final_data_structure = OrderedDict()   
        dict_blocks = self.create_blocks(data_frame)
        for block_number in dict_blocks.keys():
            sub_lists = dict_blocks[block_number]
            sublist_dfs=[]
            for each_sublist in sub_lists:
                training_df = self.create_training_dataframe(each_sublist)
                sublist_dfs.append(training_df)
            final_data_structure[block_number] = sublist_dfs
        return final_data_structure
    
    ## custom scorer created for developing corss validation strategy on reducing the false alarm rate on training dataset
    ## input : predicted output and actual output
    ## returns the false alarm rate or 0 if none
    def scorer(self,predict_output_df,actual_df):
        try:
            tn, fp, fn, tp = confusion_matrix(actual_df, predict_output_df).ravel()
            false_alarm = fp / (fp+ tp)
            return false_alarm    
        except ValueError:
            return 0
    
    ## function used to calculate the metrics required and print the result on screen. Also generates a graph 
    # input : predicted output and user no 
    # output : prints the metrics on console and saves graph in the source code folder 
    def calculate_testing_metrics(self,predict_output_df , user_no):
        predict_output_df.to_csv('predicted_output.csv')
        outputdf = pd.read_csv('actual_output.csv')
        actual_df = self.testing_actual_values
        actual_df.to_csv('actual-output.csv')
        tn, fp, fn, tp = confusion_matrix(actual_df, predict_output_df).ravel()
        print(tn,fp,fn,tp)
        accuracy_rate = (tn + tp)/(tn+tp+fn+fp)
        misses = (fn+fp)/(tn+tp+fn+fp)
        false_alarm = fp / (fp+ tp)
        print("accuracy rate is", accuracy_rate)
        print("misses rate is ", misses)
        print("false alarm rate is", false_alarm)
        plotframe = pd.concat([actual_df, predict_output_df], axis = 1)
        plotframe.columns = ['actual value','predicted value']
        plotframe.plot(kind = 'line')
        filename = os.path.join('plot_' + str(user_no) + '.png')
        plt.savefig(filename)
        return false_alarm
    
    def run_pca_test(self, all_data_frames):
        
        tsvd = TruncatedSVD(n_components= self.n_components)
        all_data_frames = tsvd.fit_transform(all_data_frames)
        print("The variance in data explained by 50 components is ",sum(tsvd.explained_variance_ratio_))
        return pd.DataFrame(all_data_frames)
    
    ## Function to build the model along with grid search for tuning hyper parameters with cross validation on false alarm rate
    # input : final data structure
    # output : returns SVM model
    def create_model_svm(self,final_data_structure):
        id_columns = list(self.database.values())
        id_columns = sorted(id_columns)
        all_data_frames = pd.DataFrame(columns= id_columns)
        for block_id , sub_block_list in final_data_structure.items():
            #print(sub_block_list)
            for df in sub_block_list:
                try:
                    all_data_frames = all_data_frames.append(df, ignore_index = True)
                except TypeError:
                    print("final data structure is " , final_data_structure )
                    raise SystemExit
        
        #*************** code for dimensionality reduction using PCA and tuning the SVM using and grid search***********
        all_data_frames = self.run_pca_test(all_data_frames)
        #print(all_data_frames.head(10))
        loss  = make_scorer(self.scorer, greater_is_better=False)
        #print(get_scorer(loss))   
        actual_df= pd.DataFrame([1]*len(all_data_frames))     
        clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma )
        parameters = {'nu': np.arange(0.0001,0.01, 0.0002)}
        print("building the model")
        grid = RandomizedSearchCV(clf, parameters, cv = 2, scoring = loss)
        model_svm2 = grid.fit(all_data_frames , actual_df)
        print("completed building the model")
        model_svm3 = model_svm2.best_estimator_
        print(model_svm2.best_params_)
        #clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma )
        #model_svm = clf.fit(all_data_frames)
        return model_svm3      
     
    # input : final metrics strucutre 
    # output: prints the block id, sub block id and the indexes at which bad blocks are found 
    def calculate_statistics(self,metrics_structure):
        predict_output_df =  pd.DataFrame([0]*len(metrics_structure))
        total_bad_blocks = 0
        total_blocks_passed = len(metrics_structure)
        for block_id , result_list in metrics_structure.items():
            num_bad_rows = result_list.count(-1)
            if(num_bad_rows > self.threshold):
                total_bad_blocks += 1
                predict_output_df.ix[block_id/100 ] = 1
                print("bad block of rows",(block_id, block_id+100))
        print("total number of blocks passed",total_blocks_passed, "total number of bad blocks", total_bad_blocks)
        abnormality_rate = (total_bad_blocks / total_blocks_passed)*100
        print("abnormality rate is ", abnormality_rate)                
        #print(predict_output_df.ix[60:63] )
        return predict_output_df
    
    # input : model and the data structure
    # output : Prints the model metrics on console  
    def model_metrics(self, type, user_no , model_svm, final_data_structure):
        metrics_structure = OrderedDict()
        arr = np.empty((0,len(self.database)), int)
        for block_id , df_list in final_data_structure.items():
                fifty_array = []
                for block in df_list:
                    with open(os.path.join(type + '_df' + str(user_no) + '.csv'), 'a') as f:
                        block.to_csv(f, header=False)
                        arr = np.vstack([arr,block])  
        arr = self.run_pca_test(arr)
        #print("array after pca in metrics" , arr[1])
        ypred = model_svm.predict(arr)
        ypred = [int(x) for x in  ypred]
        #print(len(ypred))
        block_no = 0
        for i in range(0, len(ypred),self.num_sub_blocks):
            metrics_structure[block_no] = ypred[i:i+self.num_sub_blocks]
            block_no +=100
        #print(metrics_structure)   
        predict_output_df = self.calculate_statistics(metrics_structure)              
        return dict(metrics_structure) , predict_output_df    
    
    def return_merged_ranges(self,block_numbers):
        block_numbers = [x*100 for x in block_numbers]
        new_list = []
        current_tuple = (block_numbers[0],block_numbers[0]+100)
        for i in range(len(block_numbers)-1):
            if(block_numbers[i] +100 == block_numbers[i+1]):
                temp = current_tuple[0]
                current_tuple = (temp, block_numbers[i+1])
            else:
                temp = current_tuple[0]
                current_tuple = (temp, block_numbers[i]+100)
                new_list.append(current_tuple)
                current_tuple=(block_numbers[i+1],block_numbers[i+1]+100)
        if(block_numbers[len(block_numbers)-1] == current_tuple[1]):
            temp = current_tuple[0]
            current_tuple = (temp, block_numbers[len(block_numbers)-1]+100)
            new_list.append(current_tuple)
        else:
            new_list.append(current_tuple)
            new_list.append((block_numbers[len(block_numbers)-1], block_numbers[len(block_numbers)-1]+100))
        return new_list    
    
    def create_test_train(self,user_no):
        outputdf = pd.read_csv('actual_output.csv')
        actual_df = outputdf.filter([str(user_no)], axis = 1)
        list_index_normal =  actual_df.loc[actual_df[str(user_no)] == 0].index.tolist()
        list_index_abnormal =  actual_df.loc[actual_df[str(user_no)] == 1].index.tolist()
        use_for_training = list_index_normal[:50]
        use_for_training= sorted(use_for_training)
        use_for_testing = list_index_normal[50:]
        use_for_testing.extend(list_index_abnormal)
        use_for_testing = sorted(use_for_testing)
        testing_actual_values = actual_df.ix[use_for_testing]
        list_merged_train = self.return_merged_ranges(use_for_training)
        list_merged_test = self.return_merged_ranges(use_for_testing)
        user_file = "User"+str(user_no)
        user_data_file = os.path.join(self.PATH, user_file)
        training_file = "training_new"+ str(user_no)
        training_data_file = os.path.join(self.PATH, training_file)
        testing_file = "testing_new" + str(user_no)
        testing_data_file = os.path.join(self.PATH, testing_file)
        with io.open(training_data_file, 'w') as f:
            [f.write(s) for (i, s) in enumerate(open(user_data_file,'r' )) if i < 5000]
            for tup in list_merged_train:
                file_lines = open(user_data_file,'r' )
                for i, word in enumerate(file_lines):
                    if(i in range(tup[0], tup[1])):
                        f.write(word)
                file_lines.close()
        f.close()
        with io.open(testing_data_file, 'w') as f:
            for tup in list_merged_test:
                file_lines = open(user_data_file,'r' )
                for i, word in enumerate(file_lines):
                    if(i in range(tup[0], tup[1])):
                        f.write(word)
        f.close()
        print("training data file is ",training_file )
        print("testing data file is ", testing_file)
        return testing_actual_values, training_file, testing_file
                                
    # input : user file 
    # output : dataframe of wordlists and counts with 1    
    def create_nparray(self,file_name,user_no):
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
        #return data_frame
    
    #input : driver program for training which takes user name as input
    # output : prints out the metrics of the training with bad blocks details
    def training_model(self, training_user, user_no):
        training_file = os.path.join(self.PATH,training_user) 
        final_data_structure = self.create_nparray(training_file,user_no)
        model_svm = self.create_model_svm(final_data_structure) 
        model_file = os.path.join('model_'+ str(user_no)+ '.svm') 
        dump(model_svm,open(model_file, 'wb'))   
        metrics_structure ,predict_output_df = self.model_metrics('training', user_no ,model_svm, final_data_structure)
    
    #input : driver program for testing which takes user name as input
    # output : prints out the metrics with bad block details
    def model_testing(self, testing_user, user_no):
        load(open('database.dict','rb'))
        testing_file = os.path.join(self.PATH, testing_user)
        testing_data_structure = self.create_nparray(testing_file, user_no)
        model_file = os.path.join('model_'+ str(user_no)+ '.svm')
        trained_model = load(open(model_file, 'rb'))
        testing_metrics_structure ,predict_output_df = self.model_metrics('testing', user_no , trained_model, testing_data_structure)
        self.calculate_testing_metrics(predict_output_df, user_no)

# creating object of training class and calling functions
trn =  Training(12)
trn.training_model(trn.training_file,12)
trn.model_testing(trn.testing_file,12)
