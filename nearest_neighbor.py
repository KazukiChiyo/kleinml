'''
Nearest Neighbor classifier implementation, trained on MNIST dataset.
Author: Kexuan Zou
Date: Mar 19, 2018
Confusion matrix:
[[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  0  0  0]
 [ 0  0  0 11  0  0  0  0  0  0]
 [ 0  0  0  0 14  0  0  0  0  0]
 [ 0  0  0  0  0  7  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  0  0 15  0  0]
 [ 0  0  0  0  0  0  0  0  2  0]
 [ 0  0  0  0  0  0  0  0  0 11]]
Accuracy: 1.0
'''

from math import sqrt
import util

class NearestNeighbors(object):
    def __init__(self, n_neighbors=3, algorithm="brute"):
        self.k = n_neighbors
        self.method = algorithm

    # bind labels to corresponding feature vectors
    def fit(self, feature, label):
        self.train_set = util.vbind(feature, label)
        return self

    # calculate euclidean distance between two points
    def euclidean(self, a, b):
    	distance = 0.0
    	for i in range(len(a) - 1):
    		distance += (a[i] - b[i])**2
    	return sqrt(distance)

    # given a feature vector, find its k nearest neighbors
    def get_neighbors(self, target):
        if self.method == "brute":
        	dists, neighbors = [], []
        	for elem in self.train_set:
        		dist = self.euclidean(elem, target)
        		dists.append((elem, dist))
        	dists.sort(key=lambda tup: tup[1])
        	for i in range(self.k):
        		neighbors.append(dists[i][0])
        	return neighbors

    # given a feature vector, predict its label with most popular labeling among its nearest neighbors
    def predict_one(self, target):
    	neighbors = self.get_neighbors(target)
    	dists = [row[-1] for row in neighbors]
    	pred = max(set(dists), key=dists.count)
    	return int(pred)

    # predict an unlabeled dataset
    def predict(self, dataset):
        pred = []
        for i in range(len(dataset)):
            val = self.predict_one(dataset[i])
            pred.append(val)
        return pred

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.load_mnist()
    model = NearestNeighbors(3)
    model.fit(train_x, train_y)
    pred = model.predict(test_x[:100])
    cm = util.confusion_matrix(test_y[:100], pred)
    print(cm)
    print(util.accuracy(cm))
