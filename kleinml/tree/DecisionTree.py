'''
Decision tree implementation, trained on iris dataset.
Author: Kexuan Zou
Date: Mar 28, 2018
Confusion matrix:
[[10  0  0]
 [ 0 11  0]
 [ 0  1  8]]
Accuracy: 0.966666666667
'''

import sys
sys.path.append('../')
import util
from math import sqrt
from random import randint

class DecisionTree(object):
    def __init__(self, splitter="best", max_depth=None, min_samples_split=2):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_size = min_samples_split

    # find the gini index given a split
    def gini(self, parts, classes):
        gini = 0.0
        n = float(sum([len(part) for part in parts]))
        for part in parts:
            if len(part) == 0: # if partition is empty
                continue;
            score = 0.0
            num_elems = float(len(part))
            for c in classes: # evaulate score for eah
                score += ([elem[-1] for elem in part].count(c)/num_elems)**2
            gini += (1.0 - score) * (num_elems/n) # weight score for each part
        return gini

    # split tree once based on one feature and a threshold value
    def split_once(self, idx, val, dataset):
        left, right = [], []
        for elem in dataset:
            if elem[idx] < val:
                left.append(elem)
            else:
                right.append(elem)
        return left, right

    # find the split point with min gini index
    def best_split(self, dataset):
        classes = self.class_labels
        best_index, best_value, min_gini, partitions = sys.maxsize, sys.maxsize, sys.maxsize, None

        # for method "best"
        if self.splitter == "best": # iterate through each feature in each feature vector and find best split
            for i in range(self.feature_dim): # for each feature
                for elem in dataset:
                    parts = self.split_once(i, elem[i], dataset)
                    cur_gini = self.gini(parts, classes)
                    if cur_gini < min_gini: # if better split point is found
                        best_index, best_value, min_gini, partitions = i, elem[i], cur_gini, parts

        # for method "random"
        elif self.splitter == "random": # randomly select sqrt(n) features and find best split
            for count in range(int(sqrt(self.feature_dim))): # for randomly selected features
                i = randint(0, self.feature_dim - 1) # generate a random i
                for elem in dataset:
                    parts = self.split_once(i, elem[i], dataset)
                    cur_gini = self.gini(parts, classes)
                    if cur_gini < min_gini: # if better split point is found
                        best_index, best_value, min_gini, partitions = i, elem[i], cur_gini, parts
        return {"index": best_index, "value": best_value, "partitions": partitions}

    # create a leaf node by choosing the most common label
    def create_leaf(self, part):
        labels = [int(elem[-1]) for elem in part]
        return max(set(labels), key=labels.count)

    def split(self, node, depth):
        left, right = node["partitions"]
        del(node["partitions"])
        if not left or not right: # if node has only one child,
            node["left"] = node["right"] = self.create_leaf(left + right)
            return;
        if self.max_depth is not None and depth >= self.max_depth: # if max depth is reached, create two leaf nodes
            node["left"], node["right"] = self.create_leaf(left), self.create_leaf(right)
            return;

        # for left partition
        if len(left) <= self.min_size: # if min_size prohibits a new split, create leaf node
            node["left"] = self.create_leaf(left)
        else: # if not, proceed to split the tree
            node["left"] = self.best_split(left)
            self.split(node["left"], depth+1)

        # for right partition
        if len(right) <= self.min_size: # if min_size prohibits a new split, create leaf node
            node["right"] = self.create_leaf(right)
        else: # if not, proceed to split the tree
            node["right"] = self.best_split(right)
            self.split(node["right"], depth+1)

    # build the decision tree
    def build_tree(self):
        root = self.best_split(self.dataset)
        self.split(root, 1) # children of root are at depth 1
        return root

    # create decision tree
    def fit(self, feature, label):
        self.feature_dim = len(feature[0])
        self.dataset = util.vbind(feature, label)
        self.class_labels = list(set(elem[-1] for elem in self.dataset)) # get unique class labels
        self.tree = self.build_tree()
        return self

    # given a feature vector, predict its label
    def predict_one(self, node, feature):
        if feature[node["index"]] < node["value"]: # if feature falls in left subtree
            if not isinstance(node["left"], dict): # if left child does not exist
                return node["left"]
            else: # if left child exists, recursively predict on left subtree
                return self.predict_one(node["left"], feature)
        else: # if feature falls in right part of tree
            if not isinstance(node["right"], dict): # if right child does not exist
                return node["right"]
            else: # if right child exists, recursively predict on right subtree
                return self.predict_one(node["right"], feature)

    # predict an unlabeled dataset
    def predict(self, dataset):
        pred = []
        for i in range(len(dataset)):
            pred.append(self.predict_one(self.tree, dataset[i]))
        return pred

    # helper function to print a decision tree
    def print_tree_helper(self, node, depth):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*'  ', (node['index']), node['value'])))
            self.print_tree_helper(node['left'], depth+1)
            self.print_tree_helper(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*'  ', node)))

    # print a decision tree
    def print_tree(self):
        node = self.tree
        self.print_tree_helper(node, 0)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_iris()
    model = DecisionTree(splitter="best", max_depth=3, min_samples_split=2)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
