'''
Gaussian naive bayes classifier implementation, trained on MNIST dataset.
Author: Kexuan Zou
Date: Mar 19, 2018
Confusion matrix:
[[ 887    1   18   32    6   11   15    2    8    0]
 [   0 1097   17   10    1    1    5    0    4    0]
 [  22   36  877   40   13    1   17    9   16    1]
 [   9   50   59  822   10    9    4   28    9   10]
 [  14   15   25    6  853   11   14   17    5   22]
 [  35   41   19  212   72  447   13   20   23   10]
 [  16   19   35   10    7   18  851    0    2    0]
 [   4   37   14   18   29    1    1  904    7   13]
 [  13  131   40  194   83   31    6   30  420   26]
 [   9   31    6   22  166    1    0  209    4  561]]
Accuracy: 0.7719
'''

import util, math

# summary helper function, evalute a {mean, sd} pair given data
def __summary(data):
    return [(util.mean(elem), util.sd(elem)) for elem in zip(*data)]

# evaluate feature set into {mean, sd} pairs
def nb_fit(feature, label, split=True):
    if split is False:
        return __summary(feature)
    else:
        classes = util.split_by_class(feature, label)
        summary = {}
        for c, elem in classes.iteritems():
            summary[c] = __summary(elem)
        return summary

# calculate the gaussian probability given mean and sd
def gaussian(x, mu, sd):
    if sd == 0:
        if x == mu:
            return 1;
        else:
            return 0;
    exponent = math.exp(-(math.pow(x-mu,2)/(2*math.pow(sd,2))))
    return (1 / (math.sqrt(2*math.pi) * sd)) * exponent

# evaluate the posteriori probability
def eval_probability(model, input, method="gaussian"):
    class_probs = {}
    if method == "gaussian":
        for c, elem in model.iteritems():
            class_probs[c] = 1
            for i in range(len(elem)):
                x = input[i]
                mu, sd = elem[i]
                class_probs[c] *= gaussian(x, mu, sd)
    return class_probs

# given a feature vector, predict its label with the maximum a posteriori
def get_label(model, feature, method="gaussian"):
    class_probs = eval_probability(model, feature, method)
    pred_label = None
    prob_max = 0.0
    for c, prob in class_probs.iteritems():
        if pred_label is None:
            pred_label = c
            prob_max = prob
        if prob > prob_max:
            pred_label = c
            prob_max = prob
    return pred_label

# predict an unlabeled dataset
def predict(model, dataset, method="gaussian"):
    pred = []
    for i in range(len(dataset)):
        pred.append(get_label(model, dataset[i], method))
    return pred

if __name__ == '__main__':
    train_set, valid_set, test_set = util.load_mnist()
    train_x, train_y = train_set[0], train_set[1]
    test_x, test_y = test_set[0], test_set[1]
    model = nb_fit(train_x, train_y)
    pred = predict(model, test_x)
    cm = util.confusion_matrix(test_y, pred)
    print(cm)
    print(util.accuracy(cm))
