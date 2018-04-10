'''
Gaussian naive bayes classifier implementation, trained on the mnist dataset.
Author: Kexuan Zou
Date: Mar 19, 2018. Revision: Apr 9, 2018.
Confusion matrix:
[[ 895    1   11   19    3   11   22    2   15    1]
 [   0 1099   13    9    1    1    5    0    7    0]
 [  19   36  836   60   13    2   29    9   24    4]
 [   6   48   40  820    8   10   13   28   14   23]
 [   7   13   16    6  768    7   21   13   11  120]
 [  35   44   15  212   55  421   17   15   52   26]
 [  17   20   24   12    6   13  856    0    9    1]
 [   3   23   13   15   33    2    1  900   13   25]
 [   9  124   22  125   49   24    7   18  539   57]
 [   5   21    7   20   86    0    0   97   11  762]]
Accuracy: 0.7896
'''

import numpy as np
import sys
sys.path.append('../')
import util

class GaussianNB(object):
    def __init__(self):
        pass

    # fit helper function, evalute a [mean, sd] pair given data
    def stat(self, classes):
        return np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] for i in classes])

    # evaluate feature set into [mean, sd] pairs
    def fit(self, X, Y):
        classes = [[x for x, t in zip(X, Y) if t == c] for c in np.unique(Y)]
        self.model = self.stat(classes)
        return self

    # calculate the gaussian probability given mean and sd
    def gaussian(self, x, mu, sd):
        if sd == 0: # when sd is 0 emulate a Dirac delta function
            if x == mu:
                return 1;
            else:
                return 0;
        exponent = np.exp(-((x - mu)**2 / (2*sd**2)))
        gaussian_prob = exponent/(np.sqrt(2*np.pi)*sd)
        if gaussian_prob == 0:
            return 0
        return np.log(gaussian_prob)

    # predict an unlabeled dataset
    def predict(self, X):
        prob_mat = [[sum(self.gaussian(i, *s) for s, i in zip(summaries, x)) for summaries in self.model] for x in X]
        return np.argmax(prob_mat, axis=1)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_mnist()
    model = GaussianNB()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
