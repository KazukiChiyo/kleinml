'''
Linear least squares with l2 regularization implementation, trained on the diabetes dataset.
Author: Kexuan Zou
Date: Apr 12, 2018.
Score: 0.55560274266907628
'''

import numpy as np
import numpy.linalg as la
import sys
sys.path.append('../')
import util

class Ridge(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    # evaluate ridge regression coefficient: w_hat = (X^T X + alpha I)^-1 X^T Y
    def fit(self, X, Y):
        X = X.astype(float)
        Y = Y.astype(float)
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        self.w = la.inv(X.T.dot(X) + self.alpha*np.identity(len(X[0]))).dot(X.T).dot(Y)
        return self
    
    # predict an unlabeled dataset
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        return X.dot(self.w)
    
    # score of the model
    def score(self, X, Y):
        return 1 - sum((self.predict(X) - Y)**2) / sum((Y - np.mean(Y))**2)
        
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_diabetes()    
    model = Ridge(alpha=0.2)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(score)
