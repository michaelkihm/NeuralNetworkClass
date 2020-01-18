"""
Function to load some common datasets 
to test the network
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np 

def loadMNIST(test_size=0.2):
    mnist = fetch_openml('mnist_784',version=1)
    X,y = mnist["data"], mnist["target"]
    y=y.astype(np.uint8)
    #one hot encoding
    n_values = np.max(y) + 1
    y=np.eye(n_values)[y]
    #split dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size)
    return (X_train, y_train, X_test, y_test)
