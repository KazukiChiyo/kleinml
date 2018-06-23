'''
General helper functions.
Author: Kexuan Zou
Date: Mar 19, 2018
'''

try:
   import cPickle as pickle
except:
   import pickle
import gzip
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# bind two np arrays vertically
def vbind(feature, label):
    if type(feature) is not np.ndarray:
        feature = np.array(feature)
    if type(label) is not np.ndarray:
        label = np.array(label)
    return np.c_[feature, label]

# load diabetes dataset
def load_diabetes(catog=True):
    diabetes = datasets.load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=.2)
    if catog == True:
        y_train = np.array([0 if y < 126 else 1 for y in y_train])
        y_test = np.array([0 if y < 126 else 1 for y in y_test])
    return x_train, y_train, x_test, y_test

# load iris data
def load_iris():
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.2)
    return x_train, y_train, x_test, y_test

# load breast cancer data
def load_breast_cancer():
    cancer = datasets.load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=.2)
    return x_train, y_train, x_test, y_test

# load digits data
def load_digits():
    digits = datasets.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.2)
    return x_train, y_train, x_test, y_test

# load mnist data
def load_mnist():
    script_dir = os.path.dirname(__file__)
    rel_path = "../data/mnist.pkl.gz"
    abs_file_path = os.path.join(script_dir, rel_path)
    f = gzip.open(abs_file_path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    return train_set[0], train_set[1], test_set[0], test_set[1]

# load a custom dataset to test rbf kernel
def load_rbf():
    dataMat, labelMat = load_txt("../data/testSetRBF.txt")
    x_train, x_test, y_train, y_test = train_test_split(dataMat, labelMat, test_size=.2)
    return x_train, y_train, x_test, y_test

def load_eruption():
    dataMat, labelMat = load_txt("../data/eruption.dat", label_type="float")
    x_train, x_test, y_train, y_test = train_test_split(dataMat[:,-1], labelMat, test_size=.2)
    return x_train.reshape(-1, 1), y_train, x_test.reshape(-1, 1), y_test

# generate a regression model with the form y = slope*x + err
def load_lm(slope, intercept, sd, n):
    x = np.random.randint(0, 100, size=n)
    err = np.random.randn(n)*sd
    x_train, x_test, y_train, y_test = train_test_split(x, x*slope + err + intercept, test_size=.2)
    return x_train.reshape(-1, 1), y_train, x_test.reshape(-1, 1), y_test

# load a dataset in a .txt file, separated by \t
def load_txt(rel_path, label_type="int"):
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, rel_path)
    max_idx = len(open(abs_file_path).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(abs_file_path)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(max_idx):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        if label_type == "int":
            labelMat.append(int(curLine[-1]))
        elif label_type == "float":
            labelMat.append(float(curLine[-1]))
    return np.array(dataMat), np.array(labelMat)

# split the feature set into groups by labels
def split_by_class(feature, label):
    assert len(feature) == len(label), "Feature vector and label dimension does not match"
    classes = {}
    for i in range(len(label)):
        if label[i] not in classes:
            classes[label[i]] = []
        classes[label[i]].append(feature[i])
    return classes

def binclass_svm_split(train_x, train_y, test_x, test_y, c1=0, c2=1):
    assert len(train_x) == len(train_y), "Training feature vector and label dimension does not match"
    assert len(test_x) == len(test_y), "Testing feature vector and label dimension does not match"
    train_classes = split_by_class(train_x, train_y)
    test_classes = split_by_class(test_x, test_y)
    train_x, train_y, test_x, test_y = [], [], [], []
    train_x = train_classes[c1] + train_classes[c2]
    train_y = [-1]*len(train_classes[c1]) + [1]*len(train_classes[c2])
    test_x = list(test_classes[c1]) + list(test_classes[c2])
    test_y = [-1]*len(test_classes[c1]) + [1]*len(test_classes[c2])
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

# get confusion matrix
def confusion_matrix(true_y, pred_y, nc=None):
    assert len(true_y) == len(pred_y), "True label set and predicted label set dimension does not match"
    if nc is None:
        true_copy = true_y
        nc = len(set(true_copy))
    cm = [[0] * nc for i in range(nc)]
    for true, pred in zip(true_y, pred_y):
        cm[true][pred] += 1
    return np.array(cm)

# true positive rate for a class
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return float(confusion_matrix[label, label]) / float(col.sum())

# true negative rate for a class
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return float(confusion_matrix[label, label]) / float(row.sum())

# overall accuracy with given confusion matrix
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return float(diagonal_sum) / float(sum_of_all_elements)

# overall accuracy with true label set and preducted label set
def accuracy_score(true_y, pred_y):
    assert len(true_y) == len(pred_y), "True label set and predicted label set dimension does not match"
    count = 0
    for i in range(len(true_y)):
        if true_y[i] == pred_y[i]:
            count += 1
    return float(count)/float(len(true_y))

# plot a 2d image array
def implot(img, dim=None):
    if type(img) is not np.ndarray:
        img = np.array(img)
    if dim is not None:
        img = img.reshape(dim)
    plt.imshow(img)
    plt.show()
