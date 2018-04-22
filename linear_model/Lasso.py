'''
Linear least squares with l1 regularization implementation (coordinate descent), trained on the eruption dataset.
Author: Kexuan Zou
Date: Apr 13, 2018.
Score: 0.837822200624
'''

import numpy as np
import sys
sys.path.append('../')
import util

class Lasso(object):
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
    def fit(self, X, Y):
        X = X.astype(float)
        Y = Y.astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        n_features = X.shape[0]
        w = np.zeros(X.shape[1])
        w[0] = np.sum(Y - np.dot(X[:,1:], w[1:]))/n_features
        for _ in range(self.max_iter):
            for i in range(1, len(w)):
                w_iter = w[:] # perform a deep copy
                w_iter[i] = 0.0
                res_i = Y - np.dot(X, w_iter)
                w_star = np.dot(X[:, i], res_i)
                threshold = self.alpha*n_features
                w[i] = self.soft_threasholding(w_star, threshold)/(X[:, i]**2).sum()
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
    train_x, train_y, test_x, test_y = util.load_eruption()
    model = Lasso(alpha=0.2)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(score)
