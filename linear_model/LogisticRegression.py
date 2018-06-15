"""
Author: Kexuan Zou
Date: Apr 23, 2018.
"""

import numpy as np

class LogisticRegression(object):
    """Logistic regression with stochastic gradient descent.
    Parameters:
    -----------
    tol: float
        Precision of the solution.
    max_iter: int, optional
        Maximum number of iterations for gradient descent.
    eta0: double, optional
        The initial learning rate.
    """
    def __init__(self, tol=0.001, max_iter=1000, eta0=1e-3):
        self.tol_2 = tol**2
        self.max_iter = max_iter
        self.eta0 = eta0

    # main train loop for stochastic gradient descent by repeatedly updating w on single steps until finished all iterations.
    def fit(self, X, y):
        X, y = np.array(X.astype(float)), np.array(y)
        X = np.column_stack([np.ones(len(X)), X])
        self.model = []
        for c in np.unique(y): # for each unique class, evaluate one-vs-rest models
            self.w = np.zeros(X.shape[1])
            y_copy = np.where(y == c, 1, -1) # if element = i, output 1, else output -1
            for i in range(self.max_iter):
                grad = self.gradient(X, y_copy)
                self.w = self.w - self.eta0*grad
                if np.sum(grad**2) <= self.tol_2:
                    break
            self.model.append((self.w, c)) # append ovr weights binded with its class label

    # calculate the gradient of logistic loss function
    def gradient(self, X, y):
        grad = np.zeros_like(self.w)
        for i in range(len(X)):
            z = -y[i]*np.dot(X[i], self.w)
            grad = grad + (-y[i]*X[i])*(np.exp(z))/(1 + np.exp(z))
        return grad

    # return the maximum-likelihood class label for a single element
    def predict_one(self, x):
        return max((1.0/(1.0 + np.exp(-np.dot(x, w))), c) for w, c in self.model)[1]

    # predict the class labels for an entire dataset
    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_one(x) for x in np.column_stack([np.ones(len(X)), X])])
