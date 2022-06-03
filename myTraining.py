import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def akhi_split(data,ration):
    np.random.seed(42)
    shuffed = np.random.permutation(len(data))
    test_set_size = int(10*42 )
    train_indices = shuffed [:test_set_size]
    test_indices = shuffed [test_set_size:]
    return data.iloc[test_indices],data.iloc[train_indices]
    
if __name__ == '__main__':
    #read the data
   df = pd.read_csv('akhi.csv')
   train,test = akhi_split(df,0.2)
   X_train = train[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()
   X_test = test[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()

   Y_train = train[['infectionprob']].to_numpy().reshape(2155)
   Y_test = test[['infectionprob']].to_numpy().reshape(420)

   clf = LogisticRegression()
   clf.fit(X_train,Y_train)

   # open a file, where you ant to store the data
   file = open('model.pkl', 'wb')

   # dump information to that file
   pickle.dump(clf, file)
   file.close()
   
