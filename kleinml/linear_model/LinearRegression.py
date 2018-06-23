"""
Author: Kexuan Zou
Date: Apr 12, 2018.
Score: 0.7918898007013352
"""

import numpy as np
import numpy.linalg as la

class LinearRegression(object):
    """Linear model with least squares implementation.
    Attributes:
    -----------
    w: array
        Estimated weight of the linear model.
    """
    def __init__(self):
        pass

    # evaluate linear regression coefficient: w_hat = (X^T X)^-1 X^T y
    def fit(self, X, y):
        X, y = np.array(X.astype(float)), np.array(y.astype(float))
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        self.w = la.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    # predict an unlabeled dataset
    def predict(self, X):
        X = np.array(X)
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        return X.dot(self.w)

    # score of the model
    def score(self, X, y):
        X, y = np.array(X), np.array(y)
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)
