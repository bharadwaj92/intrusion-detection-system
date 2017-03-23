import csv
import pandas as pd
import numpy as np
import io
import glob
import os
from sklearn import svm
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
    def __init__(self ):
        self.PATH = r'C:\Users\bharadwaj\Desktop\masquerade-data'
        self.block_size = 100
        self.probe_size = 10
        self.nu = 0.01
        self.kernel = "rbf"
        self.gamma = 0.0001
        self.database = self.create_database()
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
    
    def scorer(self,predict_output_df,actual_df):
        try:
            tn, fp, fn, tp = confusion_matrix(actual_df, predict_output_df).ravel()
            false_alarm = fp / (fp+ tp)
            return false_alarm    
        except ValueError:
            return 0
    
    def create_model_svm(self,final_data_structure):
        loss  = make_scorer(self.scorer, greater_is_better=False) 
        actual_df= pd.DataFrame([-1]*len(final_data_structure))     
        clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma, tol= 0.0001 )
        print("building the model")
        params = {'nu' : np.arange(0.001 , 0.01, 0.002)}
        grid = GridSearchCV(clf, param_grid= params, cv = 10, scoring = loss)
        model_svm2 = grid.fit(final_data_structure, actual_df)
        print("completed building the model")
        model_svm3 = model_svm2.best_estimator_
        print(model_svm2.best_params_, model_svm2.best_estimator_)
        #self.nu = model_svm2.best_params_['nu']
        #clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma )
        #model_svm = clf.fit(final_data_structure)
        return model_svm3
    
    def create_matrix(self,block_arr):
        l = len(self.database)
        corr_array = np.zeros((l,l))
        for i in range(0, len(block_arr)):
            temp_arr = block_arr[i:i+self.probe_size+1]
            visited= []
            for j in range(1,len(temp_arr)):
                if temp_arr[j] not in visited:
                    corr_array[self.database[temp_arr[0]]][self.database[temp_arr[j]]] += temp_arr[1:].count(temp_arr[j])
                    visited.append(temp_arr[j])
        #with io.open('correlation.csv','a') as f:
        #    for item in corr_array:
        #        writer = csv.writer(f, lineterminator='\n')
        #        writer.writerow(item)        
        return corr_array.flatten()
        
    def perform_pca(self,df):
        tsvd = TruncatedSVD(n_components= self.n_components)
        all_data_frames = tsvd.fit_transform(df)
        print("The variance in data explained by 50 components is ",sum(tsvd.explained_variance_ratio_))
        return pd.DataFrame(all_data_frames)
    
    # input : user file 
    # output : dataframe of wordlists and counts with 1    
    def create_nparray(self,t, file_name):
        word_list = []
        l = len(self.database)
        arr = np.empty((0,l*l), int)
        with io.open(file_name, 'r') as file_lines1:
            for line in file_lines1:
                word = line.strip().lower()
                word_list.append(word)
        for i in range(0,len(word_list),self.block_size):
            block_arr = word_list[i:i+self.block_size]
            corr_flat_arr = self.create_matrix(block_arr)
            arr = np.vstack([arr,corr_flat_arr])        
        if(t == 'training'):
            init_training_df = pd.DataFrame(arr)
            #init_training_df.to_csv('training_df.csv')
            training_df = self.perform_pca(init_training_df)
            return training_df
        else:
            init_testing_df = pd.DataFrame(arr)
            #init_testing_df.to_csv('testing_df.csv')
            testing_df = self.perform_pca(init_testing_df)
            return testing_df 
    
    def testing_metrics(self,t,user_no,trained_model, actual_df , df):
        predicted_df = trained_model.predict(df)
        predicted_df = [int(x) for x in predicted_df]
        print(predicted_df.count(-1) ,predicted_df.count(1) )
        print(predicted_df)
        try:
            tn, fp, fn, tp = confusion_matrix(actual_df, predicted_df).ravel()
            print(tn,fp,fn,tp)
            accuracy_rate = (tn + tp)/(tn+tp+fn+fp)
            misses = (fn+fp)/(tn+tp+fn+fp)
            false_alarm = fp / (fp+ tp)
            print("accuracy rate is", accuracy_rate)
            print("misses rate is ", misses)
            print("false alarm rate is", false_alarm)
            
        except ValueError:
            print("the confusion matrix is", confusion_matrix(actual_df, predicted_df))
            
        #plotframe = pd.concat([actual_df, predicted_df], axis = 1)
        #plotframe.columns = ['actual value','predicted value']
        #plotframe.plot(kind = 'line')
        #filename = os.path.join('plot_' + str(user_no) + '.png')
        #plt.savefig(filename)
        
        
    #input : driver program for training which takes user name as input
    # output : prints out the metrics of the training with bad blocks details
    def training_model(self, training_user, user_no):
        training_file = os.path.join(self.PATH,training_user) 
        training_df = self.create_nparray('training', training_file)
        model_svm = self.create_model_svm(training_df) 
        model_file = os.path.join('model_'+ str(user_no)+ '.svm') 
        dump(model_svm,open(model_file, 'wb'))
        actual_df= pd.DataFrame([1]*len(training_df))   
        self.testing_metrics('training', user_no ,model_svm,actual_df,training_df)
    
    #input : driver program for testing which takes user name as input
    # output : prints out the metrics with bad block details
    def model_testing(self, testing_user, user_no):
        load(open('database.dict','rb'))
        testing_file = os.path.join(self.PATH, testing_user)
        testing_df = self.create_nparray('testing',testing_file)
        model_file = os.path.join('model_'+ str(user_no)+ '.svm')
        trained_model = load(open(model_file, 'rb'))
        outputdf = pd.read_csv('actual_output.csv')
        actual_df = [-1 if x== 0 else 1 for x in outputdf[str(user_no)] ]
        self.testing_metrics('testing', user_no , trained_model , actual_df, testing_df)
        

# creating object of training class and calling functions
trn =  Training()
trn.training_model('user43_train', 43)
trn.model_testing('user43_test', 43)
