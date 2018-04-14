'''
Linear least squares with l1 regularization implementation, trained on the diabetes dataset.
Author: Kexuan Zou
Date: Apr 12, 2018.
Score: 
'''

import numpy as np
import sys
sys.path.append('../')
import util

class Lasso(object):
    def __init__(self, alpha=1.0, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
    
    def adjust(self, x, threshold):
        if x > 0 and threshold < abs(x):
            return x - threshold
        elif x < 0 and threshold  < abs(x):
            return x + threshold
        else:
            return 0

    def fit(self, X, Y):
        n_features = X.shape[0]
        X = np.column_stack([np.ones(len(X)), X])
        w = np.zeros(X.shape[1])
        w[0] = np.sum(Y - np.dot(X[:,1:], w[1:]))/n_features
        for _ in range(self.max_iter):
            for i in range(1, len(w)):
                w_iter = w[:] # perform a deep copy
                w_iter[0] = 0.0
                err_i = Y - np.dot(X, w_iter)
                x = np.dot(X[:, i], err_i)
                threshold = self.alpha*n_features
                w[i] = self.adjust(x, threshold)/(X[:, i]**2).sum()
                w[0] = np.sum(Y - np.dot(X[:,1:], w[1:]))/n_features
        self.w = w
        return self
    
    # predict an unlabeled dataset
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        return X.dot(self.w)
    
    # score of the model
    def score(self, X, Y):
        return 1 - sum((self.predict(X) - Y)**2) / sum((Y - np.mean(Y))**2)
            
if __name__ == '__main__':
    X = np.array([[1], [2], [3], [4], [5]])
    Y = np.array([1, 2, 3, 4, 5])
    model = Lasso(alpha=1.0)
    model.fit(X, Y)
    print(model.w)
