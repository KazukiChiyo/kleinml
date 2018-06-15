"""
Author: Kexuan Zou
Date: Apr 13, 2018.
Score: 0.837822200624
"""

import numpy as np

class Lasso(object):
    """Linear least squares with l1 regularization implementation (coordinate descent).
    Parameters:
    -----------
    alpha: float
        Strength of regularizaton and feature shrinkage.
    max_iter: int, optional
        Maximum number of iterations for gradient descent.
    """
    def __init__(self, alpha=1.0, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter

    # apply non-linear soft thresholding
    def soft_threasholding(self, w, threshold):
        if w > 0 and threshold < abs(w):
            return w - threshold
        elif w < 0 and threshold  < abs(w):
            return w + threshold
        else:
            return 0

    # evaluate w by coordinate descent
    def fit(self, X, y):
        X, y = np.array(X.astype(float)), np.array(y.astype(float))
        X = np.column_stack([np.ones(len(X)), X])
        n_features = X.shape[0]
        w = np.zeros(X.shape[1])
        w[0] = np.sum(y - np.dot(X[:,1:], w[1:]))/n_features
        for _ in range(self.max_iter):
            for i in range(1, len(w)):
                w_iter = w[:] # perform a deep copy
                w_iter[i] = 0.0
                res_i = y - np.dot(X, w_iter)
                w_star = np.dot(X[:, i], res_i)
                threshold = self.alpha*n_features
                w[i] = self.soft_threasholding(w_star, threshold)/(X[:, i]**2).sum()
                w[0] = np.sum(y - np.dot(X[:,1:], w[1:]))/n_features
        self.w = w
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
