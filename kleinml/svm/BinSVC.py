"""
Author: Kexuan Zou
Date: Apr 1, 2018
"""

import numpy as np

class BinSVC(object):
    """Binary support vector machine using Platt's SMO algorithm.
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
    Attributes:
    -----------
    idx: array
        Indices of support vectors.
    sv_x: array
        Support vectors.
    sv_y: array
        Labels of support vectors.
    alphas: array
        Coefficients of the support vector in the decision function.
    b: float
        Interrcept of the decision function.
    """
    def __init__ (self, C=1.0, kernel="rbf", degree=3, gamma="auto", coef0=0.0, tol=0.01, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    # evaluate dual problem using SMO algorithm, evaulate weight of svm
    def fit(self, X, y):
        self.X = np.matrix(X)
        self.y = np.matrix(y).T
        self.m = np.shape(self.y)[0]
        if self.gamma == "auto": # if auto, 1 / n_features will be used
            self.gamma = 1.0/self.m
        self.alphas = np.matrix(np.zeros((self.m, 1)))
        self.b = 0
        self.cache = np.matrix(np.zeros((self.m, 2))) # first column is a flag bit stating whether cache is valid
        self.k = np.matrix(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:,i] = self.kernel_transform(self.X, self.X[i,:])
        self.smo()
        self.sv_x, self.sv_y, self.alphas = self.get_svs()
        return self

    # apply kernel transformations to input data
    def kernel_transform(self, X, A):
        m,n = np.shape(X)
        K = np.matrix(np.zeros((m,1)))
        if self.kernel == "lin": # linear kernel
            K = X * A.T
        elif self.kernel == "rbf": # rbf kernel
            for j in range(m):
                deltaRow = X[j,:] - A
                K[j] = deltaRow*deltaRow.T
            K = np.exp(-self.gamma*K)
        elif self.kernel == "poly": # polynomial kernel
            K = (self.coef0 + np.inner(X, A)) ** self.degree
        elif self.kernel == "sigmoid": # sigmoid kernel
            K = np.tanh(self.gamma * np.dot(X, A) + self.coef0)
        return K

    # loss function for a given alpha
    def error_k(self, k):
        y_hat = float(np.multiply(self.alphas, self.y).T*self.k[:,k] + self.b)
        error = y_hat - float(self.y[k])
        return error

    # randomly select j that is within range and is not equal to i
    def rand_j(self, i):
        j = i
        while (j == i):
            j = int(np.random.uniform(0,self.m))
        return j

    # truncate alpha value with higher and lower bound
    def truncate_alpha(self, aj, hi, lo):
        if aj > hi:
            aj = hi
        if lo > aj:
            aj = lo
        return aj

    # given first alpha i, find second alpha j so that step cost is maximized
    def select_j(self, i, err_i):
        max_k, max_delta_err, err_j = -1, 0, 0
        self.cache[i] = [1, err_i]  # find alpha that gives the max_delta_err
        cache_list = np.nonzero(self.cache[:,0].A)[0] # list all valid caches
        if (len(cache_list)) > 1: # if not at first iteration, select k from cache lsit
            for k in cache_list:   # iterate through valid caches and find max_delta_err
                if k == i: # collision, skip current iteration
                    continue
                err_k = self.error_k(k)
                delta_e = abs(err_i - err_k)
                if (delta_e > max_delta_err): # if higher step size is found, update all values
                    max_k, max_delta_err, err_j = k, delta_e, err_k
            return max_k, err_j
        else: # at first iteration, pick j randomly
            j = self.rand_j(i)
            err_j = self.error_k(j)
        return j, err_j

    # set valid cache, and update error_k in in cache
    def update_err_k(self, k):
        err_k = self.error_k(k)
        self.cache[k] = [1, err_k]

    # given alpha_i, find alpha_j and update the pairs
    def update_pair(self, i):
        err_i = self.error_k(i)
        if ((self.y[i]*err_i < -self.tol) and (self.alphas[i] < self.C)) or ((self.y[i]*err_i > self.tol) and (self.alphas[i] > 0)):
            j, err_j = self.select_j(i, err_i) # select best j given i
            old_i = self.alphas[i].copy()
            old_j = self.alphas[j].copy()
            if (self.y[i] != self.y[j]):
                lo = max(0, self.alphas[j] - self.alphas[i])
                hi = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                lo = max(0, self.alphas[j] + self.alphas[i] - self.C)
                hi = min(self.C, self.alphas[j] + self.alphas[i])
            if lo==hi:
                return 0
            eta = 2.0 * self.k[i,j] - self.k[i,i] - self.k[j,j]
            if eta >= 0:
                return 0
            self.alphas[j] -= self.y[j]*(err_i - err_j)/eta
            self.alphas[j] = self.truncate_alpha(self.alphas[j],hi,lo)
            self.update_err_k(j) # update error_k for j in the cache
            if (abs(self.alphas[j] - old_j) < 0.00001):
                return 0
            self.alphas[i] += self.y[j]*self.y[i]*(old_j - self.alphas[j]) # update i by the same amount as j
            self.update_err_k(i) # update error_k for i in the cache
            b1 = self.b - err_i- self.y[i]*(self.alphas[i] - old_i)*self.k[i,i] - self.y[j]*(self.alphas[j] - old_j)*self.k[i,j]
            b2 = self.b - err_j- self.y[i]*(self.alphas[i] - old_i)*self.k[i,j] - self.y[j]*(self.alphas[j] - old_j)*self.k[j,j]
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2)/2.0
            return 1
        else:
            return 0

    # in each iteration, solve the svm dual problem and update alpha pairs
    def smo(self):
        iter = 0
        entire_flag = True
        pair_update_flag = 0
        while (iter < self.max_iter) and ((pair_update_flag > 0) or (entire_flag)):
            pair_update_flag = 0
            if entire_flag: # go over all alphas
                for i in range(self.m):
                    pair_update_flag += self.update_pair(i)
                iter += 1
            else: # go over non-bound alphas
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    pair_update_flag += self.update_pair(i)
                iter += 1
            if entire_flag:
                entire_flag = False #toggle entire set loop
            elif (pair_update_flag == 0):
                entire_flag = True

    # get all support vectors, their corresponding labels and alphas
    def get_svs(self):
        self.idx = np.nonzero(self.alphas.A>0)[0]
        return self.X[self.idx], self.y[self.idx], self.alphas[self.idx]

    # given a feature vector, predict its label
    def predict_one(self, x):
        tarmat = np.matrix(x)
        k = self.kernel_transform(self.sv_x, tarmat)
        pred_mat = k.T * np.multiply(self.sv_y, self.alphas) + self.b
        return np.sign(pred_mat[0,0]).astype(int)

    # predict an unlabeled dataset
    def predict(self, X):
        pred = []
        for i in range(len(X)):
            val = self.predict_one(X[i])
            pred.append(val)
        return pred
