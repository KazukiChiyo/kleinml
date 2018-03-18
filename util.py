import cPickle, gzip, math
import numpy as np

# calculate mean
def mean(data):
    return sum(data)/float(len(data))

# calculate sd
def sd(data):
    mu = mean(data)
    var = sum([pow(x-mu,2) for x in data])/float(len(data)-1)
    return math.sqrt(var)

# load mnist data
def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

# split the feature set into groups by labels
def split_by_class(feature, label):
    assert len(feature) == len(label), "Feature vector and label dimension does not match"
    classes = {}
    for i in range(len(label)):
        if label[i] not in classes:
            classes[label[i]] = []
        classes[label[i]].append(feature[i])
    return classes

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
