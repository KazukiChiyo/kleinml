'''
Stochastic Gradient Descent on linear regression, trained on the eruption dataset.
Author: Kexuan Zou
Date: Apr 17, 2018.
Score: 0.544482300731
'''

import numpy as np
import sys
sys.path.append('../')
import util

class SGDRegressor(object):
    def __init__(self, shuffle=True, max_iter=1000, batch_size=16, learning_rate="constant", eta0=0.001, decay=50.0, power_t=0.5):
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.decay = decay
        self.power_t = power_t

    # main training loop for stochastic gradient descent by repeatedly updatinng w on single steps until finished all iterations or batches.
    def fit(self, X, Y):
        X = X.astype(float)
        Y = Y.astype(float)
        self.w = np.zeros((X.shape[1]+1, self.batch_size)) # weight plus intercept term, duplicated in column4
        for i in range(self.max_iter):
            if self.shuffle: # if data is shuffled
                shuffle_set = util.vbind(X, Y) # bind X and Y before shuffling
                np.random.shuffle(shuffle_set)
                X_shuffle, Y_shuffle = shuffle_set[:,:-1], shuffle_set[:,-1:]
            X_batch = X_shuffle[i*self.batch_size:min((i + 1)*self.batch_size, X_shuffle.shape[0])]
            Y_batch = Y_shuffle[i*self.batch_size:min((i + 1)*self.batch_size, Y_shuffle.shape[0])]
            if len(X_batch) != 0:
                self.update_step(X_batch, Y_batch, i)
        self.w = self.w[:,0]
        return self

    # compute the gradient of loss with respect to w g[L(t)]
    def gradient(self, y_hat, Y):
        return np.dot(np.transpose(self.X), y_hat - Y)

    # update w on a single step: w(t+1) = w(t) - eta*g[L(t)]/n
    def update_step(self, X, Y, t):
        self.X = np.column_stack([np.ones(len(X)), X])
        y_hat = self.X.dot(self.w)
        grad = self.gradient(y_hat, Y)
        if self.learning_rate == "constant":
            eta = self.eta0
        elif self.learning_rate == "optimal":
            eta = self.eta0/(1. + self.decay*t)
        elif self.learning_rate == "invscaling":
            eta = self.eta0 / pow(t, self.power_t)
        self.w = self.w - eta*1.0/self.batch_size*grad

    # predict an unlabeled dataset
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X]) # append ones as intercept
        return X.dot(self.w)

    # score of the model
    def score(self, X, Y):
        return 1 - sum((self.predict(X) - Y)**2) / sum((Y - np.mean(Y))**2)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_eruption()
    model = SGDRegressor(batch_size=4, learning_rate="constant", eta0=0.01)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(score)
