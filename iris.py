import numpy as np
#import matplotlib.pyplot as plt
#import sklearn
import pandas as pd



df = pd.read_csv("Iris.csv")
#print(df.head())
Classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
X = df.iloc[:,0:4]
X = np.array(X , dtype = float)
#print(X.shape)
Y = df.iloc[:,5]
Y = np.row_stack(Y)
Y = Y[:,0]
#Y.shape = (len(Y),1)

Y1 = (Y == Classes[0])
Y1 = Y1.astype(int)

Y2 = (Y == Classes[1])
Y2 = Y2.astype(int)

Y3 = (Y == Classes[2])
Y3 = Y3.astype(int)

Y = np.stack([Y1,Y2,Y3],axis = 1)
#print(Y[95:100])
print(df.shape[0])
nh = 2
def initialise_param(ni, no, nh) :
    """
    ni = size of input layer
    no = size of output layer
    nh = size of hidden layer
    
    returns :
        W1 , b1 , W2 , b2 
    """
    W1 = np.random.randn(nh,ni)* 0.01
    b1 = np.zeros((nh,1))
    W2 = np.random.randn(no,nh)*0.01
    b2 = np.zeros((no,1))
    
    assert(W1.shape == (nh, ni))
    assert(b1.shape == (nh, 1))
    assert(W2.shape == (no, nh))
    assert(b2.shape == (no, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters  
    
