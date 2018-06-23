"""
Author: Kexuan Zou
Date: Apr 27, 2018.
"""

import numpy as np
import sys
sys.path.append('../')
from ..utils import load_data

class LinearSVC(object):
    """Linear support vector machine with stochastic gradient descent.
    Parameters:
    -----------
    C: float, optional
        Penalty parameter C of the error term.
    shuffle: boolean
        Whether or not the training data should be shuffled after each epoch.
    max_iter: int, optional
        Maximum number of iterations for gradient descent.
    tol: float
        Precision of the solution.
    batch_size: int
        Size of the batch at each iteration for mini-batch SGD.
    learning_rate: string, optional
        The learning rate schedule:
        "constant": eta = eta0
        "optimal": eta = C/t
    eta0: double, optional
        The initial learning rate.
    """
    def __init__(self, C=1.0, shuffle=True, max_iter=1000, tol=0.0001, batch_size=16, learning_rate="optimal", eta0=0.01):
        self.weight_decay = 1./C
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.tol_2 = tol**2
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eta0 = eta0

    # main training loop using stochastic gradient descent, update w in max_iter number of small steps.
    def fit(self, X, y):
        X, y = np.array(X.astype(float)), np.array(y)
        self.unique_labels = np.unique(y)
        self.model = []
        for c in self.unique_labels: # for each unique class, evaluate one-vs-rest models
            self.w = np.random.uniform(0, 1, (X.shape[1]+1, 1)) # weight plus intercept term
            y_copy = np.where(y == c, 1, -1) # if element = i, output 1, else output -1
            for i in range(self.max_iter):
                if self.shuffle: # if data is shuffled
                    shuffle_set = load_data.vbind(X, y_copy) # bind X and Y before shuffling
                    np.random.shuffle(shuffle_set)
                    X_shuffle, y_shuffle = shuffle_set[:,:-1], shuffle_set[:,-1:]
                X_batch = X_shuffle[i*self.batch_size:min((i + 1)*self.batch_size, X_shuffle.shape[0])]
                y_batch = y_shuffle[i*self.batch_size:min((i + 1)*self.batch_size, y_shuffle.shape[0])]
                if len(X_batch) != 0:
                    update = self.update_step(X_batch, y_batch, i)
                    if not update:
                        break
            self.w = self.w.flatten() # flatten w to a 1d array
            self.model.append((self.w, c)) # append ovr weights binded with its class label
        return self

    # compute the gradient of loss with respect to w g[L(t)+R]
    def gradient(self, y_hat, y):
        loss_grad = np.zeros_like(self.w) # loss function component
        reg_grad = self.w*self.weight_decay # regulization component
        for i in range(len(y)):
            if y[i]*self.X[i].dot(self.w) < 1:
                loss_grad -= (y[i]*self.X[i]).reshape(loss_grad.shape)
        return loss_grad + reg_grad

    # update w on a single step: w(t+1) = w(t) - eta*g[L(t)]/n
    def update_step(self, X, y, t):
        self.X = np.column_stack([np.ones(len(X)), X])
        y_hat = self.X.dot(self.w)
        grad = self.gradient(y_hat, y)
        if np.sum(grad**2) <= self.tol_2: # if gradient change is less than tolerance, the desent process converges
            return False
        if self.learning_rate == "optimal":
            eta = 1./(self.weight_decay*(t+1))
        elif self.learning_rate == "constant":
            eta = self.eta0
        self.w = self.w - eta*grad/self.batch_size
        return True

    def score(self, X):
        scores = []
        for w, c in self.model:
            score = X.dot(w)
            scores.append(score)
        return np.array(scores).T

    def predict(self, X):
        X = np.array(X)
        X = np.column_stack([np.ones(len(X)), X]) # insert ones as intercept
        scores = self.score(X)
        return self.unique_labels[np.argmax(scores, axis=1)]
