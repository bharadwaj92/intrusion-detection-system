import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from pickle import dump
from pickle import load
nu = 0.001
kernel = 'rbf'
gamma = 0.0001
def scorer(predict_output_df,y_train):
        try:
            tn, fp, fn, tp = confusion_matrix(y_train, predict_output_df).ravel()
            false_alarm = fp / (fp+ tn)
            print("leaving scoring function from try part")
            return false_alarm
            
        except ValueError:
            print("leaving scoring function from exception part")
            return 0
def calculate_testing_metrics(predict_output_df , y_test):
        print(confusion_matrix(y_test, predict_output_df))
        tn, fp, fn, tp = confusion_matrix(y_test, predict_output_df).ravel()
        print(tn,fp,fn,tp)
        accuracy_rate = (tn + tp)/(tn+tp+fn+fp)
        misses = (fn+fp)/(tn+tp+fn+fp)
        false_alarm = fp / (fp+ tn)
        print("accuracy rate is", accuracy_rate)
        print("misses rate is ", misses)
        print("false alarm rate is", false_alarm)
        return false_alarm
file_path = r'C:\Users\bharadwaj\Desktop\shuttle-unsupervised-ad.csv'
data=pd.read_csv(file_path,header=None)
data[9] = data[9].map({'o':-1 , 'n' : 1})
data_abnormal = data[data.ix[:,9] == -1]
data_normal = data[data.ix[:,9] == 1]
print(data_normal)
training_percent = 75 
testing_percent = 25
total_len_normal = len(data_normal)
data_train = data_normal.head(int(75*total_len_normal/100)).reset_index(drop = True)
outlier_train = data_abnormal[:20]
final_train_data = pd.concat([data_train,outlier_train], axis = 0,ignore_index = True)
final_train_data = final_train_data.reset_index(drop = True)
data_test = data_normal.tail(int(25*total_len_normal/100))
data_test_outlier = data_abnormal[20:]
final_test_data = pd.concat([data_test,data_test_outlier],axis=0,ignore_index=True)
final_test_data = final_test_data.reset_index(drop = True)
X_train = final_train_data.iloc[:,:-1]
print(X_train) 
y_train = final_train_data.iloc[:,-1]
print(y_train)
X_test = final_test_data.iloc[:,:-1]
print(X_test)
y_test = final_test_data.iloc[:,-1]
print(y_test)
N_y_train_count = sum((y_train == 1).astype(int))
O_y_train_count = sum((y_train == -1).astype(int))
N_y_test_count = sum((y_test == 1).astype(int))
O_y_test_count = sum((y_test == -1).astype(int))
print("train with 1", N_y_train_count)
print("train with -1",O_y_train_count)
print("test with 1", N_y_test_count)
print("test with -1",O_y_test_count)
#parameters = {'nu': np.arange(0.0001,0.01, 0.0002),'gamma': np.arange(0.0001,0.01, 0.0002), 'kernel': ['rbf', 'linear']}
#clf = svm.OneClassSVM(nu = nu, kernel = kernel ,gamma = gamma )
model_svm=svm.SVC()
#loss  = make_scorer(scorer, greater_is_better=False)
#grid = RandomizedSearchCV(clf, parameters, cv = 10, scoring = loss)
#model_shuttle = grid.fit( X_train,y_train)
#new_model = model_shuttle.best_estimator_
model_shuttle = model_svm.fit(X_train,y_train)
dump(model_shuttle,open('model.shuttle', 'wb')) 
trained_model = load(open('model.shuttle', 'rb'))
ypred = trained_model.predict(X_test).astype(int)
ypred = pd.DataFrame(ypred)
ypred.to_csv('predictoutput.csv')
y_test.to_csv('actualoutput.csv')
#print(np.unique(ypred))
#print(np.unique(y_test))
false_alarm_rate = calculate_testing_metrics(ypred, y_test)
