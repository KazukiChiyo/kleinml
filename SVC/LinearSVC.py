'''
Linear support vector machine using Platt's SMO algorithm, trained on iris dataset.
Author: Kexuan Zou
Date: Apr 1, 2018
Confusion matrix:
[[11  0  0]
 [ 0  9  0]
 [ 0  1  9]]
Accuracy: 0.966666666667
'''

from numpy import *
import sys
sys.path.append('../')
import util
import LinearBinSVC as bsvm

class LinearSVC(object):
    def __init__ (self, C=1.0, tol=0.01, max_iter=1000, multi_class="ovr"):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.multi_class = multi_class

    # fit features with their labels
    def fit(self, feature, label):
        self.unique_labels = unique(label)
        if self.multi_class == 'ovr':
            self.fit_ovr(feature, label)
        elif self.multi_class == 'ovo':
            self.fit_ovo(feature, label)

    # fit subroutine for one-vs-rest method, create a dictionary of svms, each of which is a binary svm
    def fit_ovr(self, feature, label):
        self.bin_svm = dict()
        for _, ul in enumerate(self.unique_labels):
            mod_y = array([1 if elem == ul else -1 for elem in label]) # label all others -1
            self.bin_svm[ul] = bsvm.LinearBinSVC(C=self.C, tol=self.tol, max_iter=self.max_iter).fit(feature, mod_y)

    # fit subroutine for one-vs-one method, create a dictionary of svms, each of which is a binary svm
    def fit_ovo(self, feature, label):
        self.bin_svm = dict()
        for i in range(self.unique_labels.shape[0]):
            for j in range(self.unique_labels.shape[0]):
                if i is not j:
                    new_x, new_y = [], []
                    for k in range(feature.shape[0]):
                        if label[k] == self.unique_labels[i]:
                            new_x.append(feature[k])
                            new_y.append(1) # label target class 1
                        elif label[k] == self.unique_labels[j]:
                            new_x.append(feature[k])
                            new_y.append(-1) # label its adversary class -1
                    self.bin_svm[(self.unique_labels[i], self.unique_labels[j])] = bsvm.LinearBinSVC(C=self.C, tol=self.tol, max_iter=self.max_iter).fit(new_x, new_y)

    # predict an unlabeled dataset
    def predict(self, dataset):
        if self.multi_class == 'ovr':
            return self.predict_ovr(dataset)
        elif self.multi_class == 'ovo':
            return self.predict_ovo(dataset)

    # predict subroutine for one-vs-rest method, vote the label with best score
    def predict_ovr(self, dataset):
        scores = self.scores_ovr(dataset)
        return self.unique_labels[argmax(scores, axis=1)]

    # predict subroutine for one-vs-one method, vote the label with best score
    def predict_ovo(self, dataset):
        scores = self.scores_ovo(dataset)
        return self.unique_labels[argmax(scores, axis=1)]

    # calculate score for ovr method
    def scores_ovr(self, dataset):
        scores = []
        for ul in self.unique_labels:
            score = self.bin_svm[ul].predict(dataset)
            scores.append(score)
        return array(scores).T

    # calculate socre for ovo method
    def scores_ovo(self, dataset):
        scores = []
        for i in range(self.unique_labels.shape[0]):
            score = zeros(dataset.shape[0])
            for j in range(0, self.unique_labels.shape[0]):
                if i != j:
                    score += self.bin_svm[(self.unique_labels[i], self.unique_labels[j])].predict(dataset)
            scores.append(score)
        return array(scores).T

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_iris()
    model = LinearSVC(multi_class="ovo")
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
