import csv
from collections import defaultdict
import pandas as pd
import io
import glob
import os
from sklearn import svm
from pickle import dump
from pickle import load

class Training():
    
    # constructor that initializes the parameters
    def __init__(self ):
        self.PATH = r'C:\Users\bharadwaj\Desktop\masquerade-data'
        self.threshold = 4
        self.row_threshold = 2
        self.word_count_size = 10
        self.block_size = 100
        self.sub_block_size = 50
        self.num_sub_blocks = 6
        self.nu = 0.1
        self.kernel = "rbf"
        self.gamma = 0.1
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
    # output: blcoks of size 100 and sub-blocks of size 50.
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
    def create_training_dataframe(self,sub_list_df):
        word_df_for_sub_list = []
        len_df = len(sub_list_df.axes[0])
        for i in range(0, len_df , self.word_count_size):
            word_count_dict = dict(self.database)
            temp_df = sub_list_df[i:i+self.word_count_size]
            temp_df1 = temp_df.groupby('id').sum()
            id_columns = list(word_count_dict.values())
            id_columns = sorted(id_columns)
            df = pd.DataFrame(columns= id_columns , index = [1])
            df.ix[1] = 0
            for ir in temp_df1.itertuples():
                df.ix[1][ir[0]] = ir[1]
            word_df_for_sub_list.append(df)
        return word_df_for_sub_list           
    
    # input : complete dataframe from a training/ testing dataset for modelling.
    # output: will call SVM model from here.        
    def create_data_format(self,data_frame):
        final_data_structure = defaultdict(dict)  # {block_number : {sub-block_number : list of dfs of size 10}} 
        dict_blocks = self.create_blocks(data_frame) # creating a dict of data blocks { block_id : sub_blocks of 50 size}
        for block_number in dict_blocks.keys():
            sub_lists = dict_blocks[block_number]
            for sub_list_number in range(len(sub_lists)):
                final_array_dataframes = self.create_training_dataframe(sub_lists[sub_list_number])
                final_data_structure[block_number][sub_list_number] = [final_array_dataframes] 
        return dict(final_data_structure)
    
    # input : final data structure
    # output : model of SVM 
    def create_model_svm(self,final_data_structure):
        id_columns = list(self.database.values())
        id_columns = sorted(id_columns)
        all_data_frames = pd.DataFrame(columns= id_columns)
        for block_id , sub_block_id in final_data_structure.items():
            for sub_block_id , df_list in final_data_structure[block_id].items():
                for small_block in df_list:
                    for item in small_block:
                        all_data_frames = all_data_frames.append(item, ignore_index = True)
        clf = svm.OneClassSVM(nu = self.nu, kernel = self.kernel ,gamma = self.gamma)
        model_svm = clf.fit(all_data_frames)
        return model_svm        
    
    # input : final metrics strucutre 
    # output: prints the block id, sub block id and the indexes at which bad blocks are found 
    def calculate_statistics(self,metrics_structure):
        total_bad_blocks = 0
        total_blocks_passed = len(metrics_structure)*self.num_sub_blocks
        for block_id , sub_block_id in metrics_structure.items():
            bad_block_count = 0
            for sub_block_id , result_list in metrics_structure[block_id].items():
                num_bad_rows = result_list.count(-1)
                if(num_bad_rows > self.row_threshold):
                    #print(result_list)
                    bad_block_count += 1
            if(bad_block_count > self.threshold):
                print("bad block of rows",(block_id , block_id+self.block_size))
                total_bad_blocks +=1
        print("total number of blocks passed",total_blocks_passed, "total number of bad blocks", total_bad_blocks)
        misclassificaiton_rate = total_bad_blocks / total_blocks_passed
        print("abnormality rate is ", misclassificaiton_rate*100)                
    
    # input : model and the data structure
    # output : Prints the model metrics on console  
    def model_metrics(self, x , model_svm, final_data_structure):
        metrics_structure = defaultdict(dict)
        for block_id , sub_block_id in final_data_structure.items():
            for sub_block_id , df_list in final_data_structure[block_id].items():
                tens_array = []
                for small_block in df_list:
                    for item in small_block:
                        with open(os.path.join(x + '_df' + '.csv'), 'a') as f:
                            item.to_csv(f, header=False)
                        tens_array.append(int(model_svm.predict(item)))
                metrics_structure[block_id][sub_block_id] = tens_array  
        self.calculate_statistics(metrics_structure)              
        return dict(metrics_structure)                
        
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
    def training_model(self, training_user):
        training_file = os.path.join(self.PATH,training_user) 
        final_data_structure = self.create_nparray(training_file)
        model_svm = self.create_model_svm(final_data_structure)  
        dump(model_svm,open('model.svm', 'wb'))     
        metrics_structure = self.model_metrics('training',model_svm, final_data_structure)
    
    #input : driver program for testing which takes user name as input
    # output : prints out the metrics with bad block details
    def model_testing(self, testing_user):
        load(open('database.dict','rb'))
        testing_file = os.path.join(self.PATH, testing_user)
        testing_data_structure = self.create_nparray(testing_file)
        trained_model = load(open('model.svm', 'rb'))
        testing_metrics_structure= self.model_metrics('testing', trained_model, testing_data_structure)

# creating object of training class and calling functions
trn =  Training()
#trn.training_model('train_2')
trn.model_testing('test2_user') 
