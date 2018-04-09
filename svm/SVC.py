'''
Support vector machine using Platt's SMO algorithm, trained on iris dataset.
Author: Kexuan Zou
Date: Apr 1, 2018
Confusion matrix:
[[11  0  0]
 [ 0  9  0]
 [ 0  1  9]]
Accuracy: 0.966666666667
'''

import numpy as np
import sys
sys.path.append('../')
import util
import BinSVC as bsvm

class SVC(object):
    def __init__ (self, C=1.0, kernel="rbf", degree=3, gamma="auto", coef0=0.0, tol=0.01, max_iter=1000, decision_function_shape="ovr"):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.multi_class = decision_function_shape

    # fit features with their labels
    def fit(self, feature, label):
        self.unique_labels = np.unique(label)
        if self.multi_class == 'ovr':
            self.fit_ovr(feature, label)
        elif self.multi_class == 'ovo':
            self.fit_ovo(feature, label)

    # fit subroutine for one-vs-rest method, create a dictionary of svms, each of which is a binary svm
    def fit_ovr(self, feature, label):
        self.bin_svm = dict()
        for _, ul in enumerate(self.unique_labels):
            mod_y = np.array([1 if elem == ul else -1 for elem in label]) # label all others -1
            self.bin_svm[ul] = bsvm.BinSVC(C=self.C, degree=self.degree, kernel=self.kernel, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter).fit(feature, mod_y)

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
                    self.bin_svm[(self.unique_labels[i], self.unique_labels[j])] = bsvm.BinSVC(C=self.C, degree=self.degree, kernel=self.kernel, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter).fit(new_x, new_y)

    # predict an unlabeled dataset
    def predict(self, dataset):
        if self.multi_class == 'ovr':
            return self.predict_ovr(dataset)
        elif self.multi_class == 'ovo':
            return self.predict_ovo(dataset)

    # predict subroutine for one-vs-rest method, vote the label with best score
    def predict_ovr(self, dataset):
        scores = self.scores_ovr(dataset)
        return self.unique_labels[np.argmax(scores, axis=1)]

    # predict subroutine for one-vs-one method, vote the label with best score
    def predict_ovo(self, dataset):
        scores = self.scores_ovo(dataset)
        return self.unique_labels[np.argmax(scores, axis=1)]

    # calculate score for ovr method
    def scores_ovr(self, dataset):
        scores = []
        for ul in self.unique_labels:
            score = self.bin_svm[ul].predict(dataset)
            scores.append(score)
        return np.array(scores).T

    # calculate socre for ovo method
    def scores_ovo(self, dataset):
        scores = []
        for i in range(self.unique_labels.shape[0]):
            score = np.zeros(dataset.shape[0])
            for j in range(0, self.unique_labels.shape[0]):
                if i != j:
                    score += self.bin_svm[(self.unique_labels[i], self.unique_labels[j])].predict(dataset)
            scores.append(score)
        return np.array(scores).T

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_iris()
    model = SVC(kernel="lin", decision_function_shape="ovo")
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
