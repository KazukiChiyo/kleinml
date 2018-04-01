'''
Linear support vector machine using Platt's SMO algorithm, trained on class 1 and 2 of the iris dataset.
Author: Kexuan Zou
Date: Apr 1, 2018
Confusion matrix:
[[ 8  1]
 [ 0 13]]
Accuracy: 0.954545454545
'''

from numpy import *
import util

class LinearSVC(object):
    def __init__ (self, C=1.0, tol=0.01, max_iter=1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    # calculate dual problem using SMO algorithm, evaulate weight of svm
    def fit(self, feature, label):
        self.x = matrix(feature)
        self.y = matrix(label).T
        self.m = shape(self.y)[0]
        self.alphas = matrix(zeros((self.m, 1)))
        self.b = 0
        self.cache = matrix(zeros((self.m, 2))) # first column is a flag bit stating whether cache is valid
        self.smo()
        self.eval_weight()
        return self

    # loss function for a given alpha
    def error_k(self, k):
        y_hat = float(multiply(self.alphas, self.y).T*(self.x*self.x[k,:].T) + self.b)
        error = y_hat - float(self.y[k])
        return error

    # randomly select j that is within range and is not equal to i
    def rand_j(self, i):
        j = i
        while (j == i):
            j = int(random.uniform(0,self.m))
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
        cache_list = nonzero(self.cache[:,0].A)[0] # list all valid caches
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
            eta = 2.0 * self.x[i,:]*self.x[j,:].T - self.x[i,:]*self.x[i,:].T - self.x[j,:]*self.x[j,:].T
            if eta >= 0:
                return 0
            self.alphas[j] -= self.y[j]*(err_i - err_j)/eta
            self.alphas[j] = self.truncate_alpha(self.alphas[j],hi,lo)
            self.update_err_k(j) # update error_k for j in the cache
            if (abs(self.alphas[j] - old_j) < 0.00001):
                return 0
            self.alphas[i] += self.y[j]*self.y[i]*(old_j - self.alphas[j]) # update i by the same amount as j
            self.update_err_k(i) # update error_k for i in the cache
            b1 = self.b - err_i- self.y[i]*(self.alphas[i] - old_i)*self.x[i,:]*self.x[i,:].T - self.y[j]*(self.alphas[j] - old_j)*self.x[i,:]*self.x[j,:].T
            b2 = self.b - err_j- self.y[i]*(self.alphas[i] - old_i)*self.x[i,:]*self.x[j,:].T - self.y[j]*(self.alphas[j] - old_j)*self.x[j,:]*self.x[j,:].T
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
                nonBoundIs = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    pair_update_flag += self.update_pair(i)
                iter += 1
            if entire_flag:
                entire_flag = False #toggle entire set loop
            elif (pair_update_flag == 0):
                entire_flag = True

    # evaluate weight components of the svm
    def eval_weight(self):
        m, n = shape(self.x)
        self.w = zeros((n,1))
        for i in range(m):
            self.w += multiply(self.alphas[i]*self.y[i], self.x[i,:].T)
        self.w = matrix(self.w)

    # given a feature vector, predict its label
    def predict_one(self, target):
        pred_mat = matrix(target)*matrix(self.w)+self.b
        pred = pred_mat[0, 0]
        if pred < 0:
            return -1
        else:
            return 1

    # predict an unlabeled dataset
    def predict(self, dataset):
        pred = []
        for i in range(len(dataset)):
            val = self.predict_one(dataset[i])
            pred.append(val)
        return pred

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_iris()
    train_x, train_y, test_x, test_y = util.binclass_svm_split(train_x, train_y, test_x, test_y, 1, 2)
    model = LinearSVC()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    test_y = [0 if x==-1 else x for x in test_y]
    pred = [0 if x==-1 else x for x in pred]
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
