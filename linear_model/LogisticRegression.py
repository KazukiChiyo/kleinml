'''
Logistic regression with stochastic gradient descent, trained on the diabetes dataset.
Author: Kexuan Zou
Date: Apr 23, 2018.
Confusion Matrix:
[[19 17]
 [ 7 46]]
Accuracy: 0.730337078652
'''

import numpy as np
import sys
sys.path.append('../')
import util

class LogisticRegression(object):
    def __init__(self, tol=0.001, max_iter=1000, eta0=1e-3):
        self.tol_2 = tol**2
        self.max_iter = max_iter
        self.eta0 = eta0
    
    # main train loop for stochastic gradient descent by repeatedly updating w on single steps until finished all iterations.
    def fit(self, X, Y):
        X = X.astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        self.model = []
        for c in np.unique(Y): # for each unique class, evaluate one-vs-rest models
            self.w = np.zeros(X.shape[1])
            Y_copy = np.where(Y == c, 1, -1) # if element = i, output 1, else output -1
            for i in range(self.max_iter):
                grad = self.gradient(X, Y_copy)
                self.w = self.w - self.eta0*grad
                if np.sum(grad**2) <= self.tol_2:
                    break
            self.model.append((self.w, c)) # append ovr weights binded with its class label

    # calculate the gradient of logistic loss function
    def gradient(self, X, Y):
        grad = np.zeros_like(self.w)
        for i in range(len(X)):
            z = -Y[i]*np.dot(X[i], self.w)
            grad = grad + (-Y[i]*X[i])*(np.exp(z))/(1 + np.exp(z))
        return grad

    # return the maximum-likelihood class label for a single element
    def predict_one(self, x):
        return max((1.0/(1.0 + np.exp(-np.dot(x, w))), c) for w, c in self.model)[1]

    # predict the class labels for an entire dataset
    def predict(self, X):
        return [self.predict_one(x) for x in np.column_stack([np.ones(len(X)), X])]

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_diabetes()
    model = LogisticRegression()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
