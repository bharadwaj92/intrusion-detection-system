from collections import defaultdict
import pandas as pd
import io
import glob
from sklearn import svm
from pickle import dump
from pickle import load
# Global parameters of the model and the database structure
database = {}
PATH = r'C:\Users\bharadwaj\Desktop\masquerade-data'
threshold = 2
word_count_size = 10
block_size = 100
sub_block_size = 50
num_sub_blocks = 6
testing_file_name = 'User1'
# input : all user files 
# output : dictionary of {'word' , id} used for look up purposes
def create_database():  
    word_id = 0
    files = [file for file in glob.glob(PATH + '/**/*', recursive=True)]
    for file_name in files:
        with io.open('User1', 'r') as file_lines:
            for line in file_lines:
                word = line.strip().lower()
                if (word not in database.keys()):
                    database[word] = word_id
                    word_id += 1
                else:
                    continue

# input : entire file as dataframe
# output: blcoks of size 100 and sub-blocks of size 50.
def create_blocks(data_frame): 
    dict_dataframes = {}
    len_df = len(data_frame.axes[0])
    #print(len_df)
    for i in range(0,len_df, block_size):
        sub_blocks = []
        temp_df = data_frame[i:i+block_size]
        for j in range(0, num_sub_blocks):
            sub_blocks.append(temp_df[j*10:j*10+sub_block_size])
        dict_dataframes[i] = sub_blocks
    return dict_dataframes

# input : each sublist of size 50
# output : a trainable sparse data frame of size length of database with columns 'id' and count. 
# if 'id' is not there in the 10 window span, it has count 0 , else the count is populated by window aggregation.        
def create_training_dataframe(sub_list_df):
    word_df_for_sub_list = []
    len_df = len(sub_list_df.axes[0])
    for i in range(0, len_df , word_count_size):
        word_count_dict = dict(database)
        temp_df = sub_list_df[i:i+word_count_size]
        temp_df1 = temp_df.groupby('id').sum()
        id_columns = list(word_count_dict.values())
        id_columns = sorted(id_columns)
        df = pd.DataFrame(columns= id_columns , index = [1])
        df.ix[1] = 0
        #print(df)
        for ir in temp_df1.itertuples():
            #print(ir[0] , ir[1])
            #print(df.ix[1])
            df.ix[1][ir[0]] = ir[1]
        word_df_for_sub_list.append(df)
    return word_df_for_sub_list           

# input : complete dataframe from a training/ testing dataset for modelling.
# output: will call SVM model from here.        
def create_data_format(data_frame):
    final_data_structure = defaultdict(dict)  # {block_number : {sub-block_number : list of dfs of size 10}} 
    dict_blocks = create_blocks(data_frame) # creating a dict of data blocks { block_id : sub_blocks of 50 size}
    for block_number in dict_blocks.keys():
        sub_lists = dict_blocks[block_number]
        for sub_list_number in range(len(sub_lists)):
            final_array_dataframes = create_training_dataframe(sub_lists[sub_list_number])
            #print(type(final_array_dataframes[0]))
            final_data_structure[block_number][sub_list_number] = [final_array_dataframes] 
    return dict(final_data_structure)

# input : final data structure
# output : model of SVM 
def create_model_svm(final_data_structure):
    id_columns = list(database.values())
    id_columns = sorted(id_columns)
    all_data_frames = pd.DataFrame(columns= id_columns)
    for block_id , sub_block_id in final_data_structure.items():
        for sub_block_id , df_list in final_data_structure[block_id].items():
            for small_block in df_list:
                for item in small_block:
                    all_data_frames = all_data_frames.append(item, ignore_index = True)
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    model_svm = clf.fit(all_data_frames)
    #print(model_svm)
    return model_svm        

# input : final metrics strucutre 
# output: prints the block id, sub block id and the indexes at which bad blocks are found 
def calculate_statistics(metrics_structure):
    for block_id , sub_block_id in metrics_structure.items():
        for sub_block_id , result_list in metrics_structure[block_id].items():
            num_bad_blocks = result_list.count(-1)
            #print(result_list, num_bad_blocks)
            if(num_bad_blocks > threshold):
                #print("bad block found at block_id",  block_id, "sub block", sub_block_id," and index", [i for i, x in enumerate(result_list) if x == -1], "and the number of bad min blocks is" ,num_bad_blocks )        
                print("bad block of rows", (block_id*block_size +block_id*block_size+block_size) ,"sub block range",(block_id*block_size+sub_block_id*word_count_size,block_id*block_size+sub_block_id*word_count_size +sub_block_size)  )

# input : model and the data structure
# output : Prints the model metrics on console  
def model_metrics(model_svm, final_data_structure):
    metrics_structure = defaultdict(dict)
    for block_id , sub_block_id in final_data_structure.items():
        for sub_block_id , df_list in final_data_structure[block_id].items():
            tens_array = []
            for small_block in df_list:
                for item in small_block:
                    tens_array.append(int(model_svm.predict(item)))
            #print(block_id, sub_block_id)
            metrics_structure[block_id][sub_block_id] = tens_array  
    calculate_statistics(metrics_structure)              
    return dict(metrics_structure)                
    
# input : user file 
# output : dataframe of wordlists and counts with 1    
def create_nparray(file_name):
    word_list = []
    with io.open('User1', 'r') as file_lines1:
        for line in file_lines1:
            word = line.strip().lower()
            word_id = database[word]
            word_list.append((word_id, 1))
    labels = ['id' , 'count']
    data_frame = pd.DataFrame.from_records(word_list, columns = labels) 
    final_data_structure = create_data_format(data_frame)        
    return final_data_structure

def model_testing(testing_file_name):
    load(open('database.dict','rb'))
    testing_data_structure = create_nparray(testing_file_name)
    trained_model = load(open('model.svm', 'rb'))
    testing_metrics_structure= model_metrics( trained_model, testing_data_structure)

def training_model():
    create_database()
    final_data_structure = create_nparray('User1')
    dump(database, open('database.dict','wb'))
    model_svm = create_model_svm(final_data_structure)  
    dump(model_svm,open('model.svm', 'wb'))     
    metrics_structure = model_metrics(model_svm, final_data_structure)

#model_testing(testing_file_name)
if __name__ == "__main__":
    training_model()
            