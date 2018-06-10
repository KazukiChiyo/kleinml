"""
Author: Kexuan Zou
Date: Apr 12, 2018.
Score: 0.7918898007013352
"""

import numpy as np
import numpy.linalg as la
import sys
sys.path.append('../')
import util

class LinearRegression(object):
    """Linear model with least squares implementation.
    """
    def __init__(self):
        pass

    # evaluate linear regression coefficient: w_hat = (X^T X)^-1 X^T Y
    def fit(self, X, Y):
        X = X.astype(float)
        Y = Y.astype(float)
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        self.w = la.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return self

    # predict an unlabeled dataset
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        return X.dot(self.w)

    # score of the model
    def score(self, X, Y):
        return 1 - sum((self.predict(X) - Y)**2) / sum((Y - np.mean(Y))**2)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_eruption()
    model = LinearRegression()
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(score)
