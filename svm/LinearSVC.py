'''
Linear support vector machine with stochastic gradient descent, trained on the iris dataset.
Author: Kexuan Zou
Date: Apr 27, 2018.
Confusion Matrix:
[[10  0  0]
 [ 0  3  5]
 [ 0  0 12]]
Accuracy: 0.833333333333
'''

import numpy as np
import sys
sys.path.append('../')
import util

class LinearSVC(object):
    def __init__(self, C=1.0, shuffle=True, max_iter=1000, tol=0.0001, batch_size=16, eta0=0.01):
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.tol_2 = tol**2
        self.batch_size = batch_size
        self.eta0 = eta0
        self.weight_decay = 1./C

    # main training loop using stochastic gradient descent, update w in max_iter number of small steps.
    def fit(self, X, Y):
        X = X.astype(float)
        self.model = []
        for c in np.unique(Y): # for each unique class, evaluate one-vs-rest models
            self.w = np.zeros((X.shape[1]+1, 1)) # weight plus intercept term
            Y_copy = np.where(Y == c, 1, -1) # if element = i, output 1, else output -1
            for i in range(self.max_iter):
                if self.shuffle: # if data is shuffled
                    shuffle_set = util.vbind(X, Y_copy) # bind X and Y before shuffling
                    np.random.shuffle(shuffle_set)
                    X_shuffle, Y_shuffle = shuffle_set[:,:-1], shuffle_set[:,-1:]
                X_batch = X_shuffle[i*self.batch_size:min((i + 1)*self.batch_size, X_shuffle.shape[0])]
                Y_batch = Y_shuffle[i*self.batch_size:min((i + 1)*self.batch_size, Y_shuffle.shape[0])]
                if len(X_batch) != 0:
                    update = self.update_step(X_batch, Y_batch, i)
                    if not update:
                        break
            self.w = self.w.flatten() # flatten w to a 1d array
            self.model.append((self.w, c)) # append ovr weights binded with its class label
        return self

    # compute the gradient of loss with respect to w g[L(t)+R]
    def gradient(self, y_hat, Y):
        loss_grad = np.zeros_like(self.w) # loss function component
        reg_grad = self.w*self.weight_decay # regulization component
        for i in range(len(Y)):
            if Y[i]*self.X[i].dot(self.w) < 1:
                loss_grad -= (Y[i]*self.X[i]).reshape(loss_grad.shape)
        return loss_grad + reg_grad

    # update w on a single step: w(t+1) = w(t) - eta*g[L(t)]/n
    def update_step(self, X, Y, t):
        self.X = np.column_stack([np.ones(len(X)), X])
        y_hat = self.X.dot(self.w)
        grad = self.gradient(y_hat, Y)
        if np.sum(grad**2) <= self.tol_2: # if gradient change is less than tolerance, the desent process converges
            return False
        self.w = self.w - self.eta0*1.0/self.batch_size*grad
        return True

    # return the maximum-likelihood class label for a single element
    def predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.model)[1]

    # predict the class labels for an entire dataset
    def predict(self, X):
        return [self.predict_one(x) for x in np.column_stack([np.ones(len(X)), X])]

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_iris()
    model = LinearSVC(batch_size=4)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
