import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
nu = 0.001
kernel = 'rbf'
gamma = 0.0001
def scorer(predict_output_df,y_train):
        try:
            tn, fp, fn, tp = confusion_matrix(y_train, predict_output_df).ravel()
            false_alarm = fp / (fp+ tp)
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
        false_alarm = fp / (fp+ tp)
        print("accuracy rate is", accuracy_rate)
        print("misses rate is ", misses)
        print("false alarm rate is", false_alarm)
        return false_alarm
file_path = r'C:\Users\bharadwaj\Desktop\shuttle-unsupervised-ad.csv'
data=pd.read_csv(file_path,header=None)
data[9] = data[9].map({'o':-1 , 'n' : 1})
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1],random_state= 100)
N_y_train_count = sum((y_train == 1).astype(int))
O_y_train_count = sum((y_train == -1).astype(int))
N_y_test_count = sum((y_test == 1).astype(int))
O_y_test_count = sum((y_test == -1).astype(int))
print("train with -1", N_y_train_count)
print("train with 1",O_y_train_count)
print("test with -1", N_y_test_count)
print("test with 1",O_y_test_count)
parameters = {'nu': np.arange(0.0001,0.01, 0.0002),'gamma': np.arange(0.0001,0.01, 0.0002), 'kernel': ['rbf', 'linear']}
clf = svm.OneClassSVM(nu = nu, kernel = kernel ,gamma = gamma )
loss  = make_scorer(scorer, greater_is_better=False)
grid = RandomizedSearchCV(clf, parameters, cv = 10, scoring = loss)
model_shuttle = grid.fit( X_train,y_train)
new_model = model_shuttle.best_estimator_
ypred = new_model.predict(X_test).astype(int)
ypred = pd.DataFrame(ypred)
ypred.to_csv('predictoutput.csv', 'w')
y_test.to_csv('actuatoutput.csv', 'w')
#print(np.unique(ypred))
#print(np.unique(y_test))
false_alarm_rate = calculate_testing_metrics(ypred, y_test)
