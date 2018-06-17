"""
Author: Kexuan Zou
Date: Apr 1, 2018
"""

import numpy as np
import sys
sys.path.append('../svm')
from BinSVC import BinSVC

class SVC(object):
    """C-Support vector classification using Platt's SMO algorithm.
    Parameters:
    -----------
    C: float, optional
        Penalty parameter C of the error term.
    kernel: string, optional
        Specifies the kernel type to be used in the algorithm:
        "linear": linear kernel <x, x'>
        "rbf": rbf kernel exp(-gamma|x-x'|^2)
        "poly": polynominal kernel (gamma<x, x'>+r)^d
        "sigmoid": sigmoid kernel tanh((gamma<x, x'>+r)^d)
    degree: int, optional
        Degree of the polynomial kernel ("poly").
    gamma: int, optional
        Kernel coefficient for "rbf", "poly" and "sigmoid".
    coef0: float, optional
        Independent term in "poly" and "sigmoid".
    tol: float
        Precision of the solution.
    max_iter: int, optional
        Hard limit on iterations within solver.
    decision_function_shape: string
        Whether to return a one-vs-rest ("ovr") decision function or one-vs-one ("ovo").
    Attributes:
    -----------
    idx: array
        Indices of support vectors.
    sv_x: array
        Support vectors.
    sv_y: array
        Labels of support vectors.
    """
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
    def fit(self, X, y):
        self.unique_labels = np.unique(y)
        self.idx, self.sv_x, self.sv_y = [], []
        self.bin_svm = dict()
        if self.multi_class == 'ovr':
            self.fit_ovr(X, y)
        elif self.multi_class == 'ovo':
            self.fit_ovo(X, y)
        self.sv_x = np.array(self.sv_x)
        self.sv_y = np.array(self.sv_y).flatten()

    # fit subroutine for one-vs-rest method, create a dictionary of svms, each of which is a binary svm
    def fit_ovr(self, X, y):
        for _, ul in enumerate(self.unique_labels):
            mod_y = np.array([1 if elem == ul else -1 for elem in y]) # label all others -1
            self.bin_svm[ul] = BinSVC(C=self.C, degree=self.degree, kernel=self.kernel, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter).fit(X, mod_y)
            self.idx.extend(self.bin_svm[ul].idx.tolist())
            self.sv_x.extend(self.bin_svm[ul].sv_x.tolist())
            self.sv_y.extend(self.bin_svm[ul].sv_y.tolist())

    # fit subroutine for one-vs-one method, create a dictionary of svms, each of which is a binary svm
    def fit_ovo(self, X, y):
        for i in range(self.unique_labels.shape[0]):
            for j in range(self.unique_labels.shape[0]):
                if i is not j:
                    new_x, new_y = [], []
                    for k in range(X.shape[0]):
                        if y[k] == self.unique_labels[i]:
                            new_x.append(X[k])
                            new_y.append(1) # label target class 1
                        elif y[k] == self.unique_labels[j]:
                            new_x.append(X[k])
                            new_y.append(-1) # label its adversary class -1
                    self.bin_svm[(self.unique_labels[i], self.unique_labels[j])] = BinSVC(C=self.C, degree=self.degree, kernel=self.kernel, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter).fit(new_x, new_y)
                    self.idx.extend(self.bin_svm[(self.unique_labels[i], self.unique_labels[j])].idx.tolist())
                    self.sv_x.extend(self.bin_svm[(self.unique_labels[i], self.unique_labels[j])].sv_x.tolist())
                    self.sv_y.extend(self.bin_svm[(self.unique_labels[i], self.unique_labels[j])].sv_y.tolist())

    # predict an unlabeled dataset
    def predict(self, X):
        if self.multi_class == 'ovr':
            return self.predict_ovr(X)
        elif self.multi_class == 'ovo':
            return self.predict_ovo(X)

    # predict subroutine for one-vs-rest method, vote the label with best score
    def predict_ovr(self, X):
        scores = self.scores_ovr(X)
        return self.unique_labels[np.argmax(scores, axis=1)]

    # predict subroutine for one-vs-one method, vote the label with best score
    def predict_ovo(self, X):
        scores = self.scores_ovo(X)
        return self.unique_labels[np.argmax(scores, axis=1)]

    # calculate score for ovr method
    def scores_ovr(self, X):
        scores = []
        for ul in self.unique_labels:
            score = self.bin_svm[ul].predict(X)
            scores.append(score)
        return np.array(scores).T

    # calculate socre for ovo method
    def scores_ovo(self, X):
        scores = []
        for i in range(self.unique_labels.shape[0]):
            score = np.zeros(X.shape[0])
            for j in range(0, self.unique_labels.shape[0]):
                if i != j:
                    score += self.bin_svm[(self.unique_labels[i], self.unique_labels[j])].predict(X)
            scores.append(score)
        return np.array(scores).T
