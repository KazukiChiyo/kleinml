'''
Bernoulli naive bayes classifier implementation, trained on the mnist dataset.
Author: Kexuan Zou
Date: Apr 9, 2018.
Confusion matrix:
[[ 887    0    3    7    2   41   17    1   22    0]
 [   0 1086   10    5    0    9    6    0   19    0]
 [  19    8  853   28   17    4   33   14   54    2]
 [   5   15   34  844    0   13    9   17   47   26]
 [   2    6    3    0  794    5   21    1   24  126]
 [  25   14    7  127   30  628   17    7   18   19]
 [  18   19   14    2   13   35  851    0    6    0]
 [   1   24   14    4   15    0    0  870   27   73]
 [  17   24   12   73   17   23    8    6  758   36]
 [   9   13    5    9   73    7    0   24   24  845]]
Accuracy: 0.8416
'''

import numpy as np
import sys
sys.path.append('../')
import util

class BernoulliNB(object):
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize

    # binarize each feature vector in a dataset into binary values
    def do_binarize(self, X):
        if self.binarize is not None:
            return np.where(X > self.binarize, 1, 0)
        else:
            return X

    # calculate the prior log probabilities for each class: log p(c)
    def class_log_prior(self, classes):
        return [np.log(float(len(c)) / self.n_obs) for c in classes]

    # calculate the probability for each feature given its class: p(count|class)
    def feature_prob(self, classes):
        count = np.array([np.array(c).sum(axis=0) for c in classes]) + self.alpha
        total = np.array([len(i) + 2*self.alpha for i in classes])
        return count/total[np.newaxis].T

    # fit a model with dataset and its corresponding labels
    def fit(self, X, Y):
        bin_X = self.do_binarize(X)
        self.n_obs = X.shape[0]
        classes = [[x for x, t in zip(bin_X, Y) if t == c] for c in np.unique(Y)]
        self.class_prior = self.class_log_prior(classes)
        self.model = self.feature_prob(classes)
        return self

    # calculate the log probability of each class: 1-p(count|class)
    def bernoulli(self, X):
        return [(np.log(self.model)*x + np.log(1 - self.model)*np.abs(x - 1)).sum(axis=1) + self.class_prior for x in X]

    # predict an unlabeled dataset
    def predict(self, X):
        bin_X = self.do_binarize(X)
        return np.argmax(self.bernoulli(bin_X), axis=1)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_mnist()
    model = BernoulliNB()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
