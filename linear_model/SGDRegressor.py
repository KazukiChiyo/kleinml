'''
Stochastic Gradient Descent on linear regression, trained on a simple linear model: y = 2x + 1.5.
Author: Kexuan Zou
Date: Apr 17, 2018.
Score: 0.544482300731
'''

import numpy as np
import sys
sys.path.append('../')
import util

class SGDRegressor(object):
    def __init__(self, shuffle=True, max_iter=1000, batch_size=16, learning_rate="constant", eta0=0.001, power_t=0.5):
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eta0 = eta0
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
            start = i*self.batch_size
            if (i+1)*self.batch_size <= len(X_shuffle):
                end = (i + 1)*self.batch_size
            else:
                end = len(X_shuffle)
            X_batch, Y_batch = X_shuffle[start:end], Y_shuffle[start:end]
            if len(X_batch) != 0:
                self.update_step(X_batch, Y_batch, i)
        self.w = self.w[:,0]
        return self

    # compute the gradient of loss with respect to w ∇L(t)
    def gradient(self, f, Y):
        return np.dot(np.transpose(self.X), f - Y)

    # update w on a single step: w(t+1) = w(t) − η∇L(t)/n
    def update_step(self, X, Y, t):
        self.X = np.column_stack([np.ones(len(X)), X])
        f = self.X.dot(self.w)
        grad = self.gradient(f, Y)
        if self.learning_rate == "constant":
            eta = self.eta0
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
    train_x, train_y, test_x, test_y = util.load_lm(slope=2, intercept=1.5, sd=50, n=200)
    model = SGDRegressor(batch_size=16, learning_rate="constant", eta0=1e-4)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(score)
