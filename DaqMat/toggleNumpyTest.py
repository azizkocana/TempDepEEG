import sys
import numpy as np

def testTransfer(X):	
    Y = np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24]]])    
    #Y = np.array([[1,2,3,4],[5,6,7,8]])    
    print(  X.flatten())
    print(X)
    print(Y.flatten())
    print(Y)
    return X, Y
