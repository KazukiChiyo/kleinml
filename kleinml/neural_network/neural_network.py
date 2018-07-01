import numpy as np
from .optimizers import SGD, Adagrad
from .base import SquareLoss, CrossEntropy
from ..utils import load_data, train_test
from .layers import Dense, Activation

loss_functions = {
    "square": SquareLoss,
    "cross_entropy": CrossEntropy
}

optimizers = {
    "sgd": SGD,
    "adagrad": Adagrad
}

class NeuralNetwork(object):
    """Neural network base model.
    Parameters:
    -----------
    optimizer: string, optional
        Optimizer to perform optimization problem:
        "sgd": Stochastic gradient descent
        "adagrad": Adaptive gradient descent
    learning_rate: float
        Learning rate.
    loss: string, optional
        Loss function used in the optimization problem:
        "square": Square loss
        "cross_entropy": Cross entropy
    layers: list, optional
        A list of layers used to train the neural network.
    validation_data: tuple, optional
        Dataset (X, y) used for validation.
    max_iter: int, optional
        Maximum number of iterations used in the optimizer.
    batch_size: int
        Size of the batch at each iteration.
    one_hot: boolen
        Whether label set is nominal or categorical.
    """
    def __init__(self, optimizer="adagrad", learning_rate=0.01, loss="cross_entropy", layers=None, validation_data=None, max_iter=1000, batch_size=32, one_hot=False):
        self.optimizer = optimizers[optimizer](learning_rate=learning_rate)
        self.loss_function = loss_functions[loss]()
        self.layers = []
        if layers is not None:
            for layer in layers:
                self.add(layer)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.errors = {"training": [], "validation": []}
        self.val_set = None
        if validation_data:
            val_X, val_y = validation_data
            self.val_X = val_X
            self.val_y = val_y

    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """Add a new layer to the neural notwork base model."""
        if self.layers: # layer is not the input layer
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        if hasattr(layer, "initialize"): # initialize the layer
            layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    # Forward pass of the network
    def _forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output

    # Back propagation for each layer in the network and update their weights in each layer
    def _backward(self, accum_grad):
        for layer in reversed(self.layers):
            accum_grad = layer.backward(accum_grad)

    def test_batch(self, X, y):
        y_pred = self._forward(X, training=False)
        return np.mean(self.loss_function.loss(y, y_pred))

    def train_batch(self, X, y):
        y_pred = self._forward(X)
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward(loss_grad)
        return np.mean(self.loss_function.loss(y, y_pred))

    def fit(self, X, y):
        """Fit the model to data matrix X and targets y."""
        if not self.one_hot: # if label set is not one hot
            y = load_data.one_hot(y)
        for _ in range(self.max_iter):
            batch_error = []
            for X_batch, y_batch in train_test.batch_iter(X, y, batch_size=self.batch_size):
                loss = self.train_batch(X_batch, y_batch)
                batch_error.append(loss)
            self.errors["training"].append(np.mean(batch_error))
            if self.val_set is not None:
                val_loss = self.test_batch(self.val_X, self.val_y)
                self.errors["validation"].append(val_loss)
        return self

    def predict(self, X):
        """Predict using the neural network base model."""
        y_pred = self._forward(X, training=False)
        if not self.one_hot:
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class LogisticRegression(object):
    """Logistic regression using various optimization techniques.
    Parameters:
    -----------
    optimizer: string, optional
        Optimizer to perform optimization problem:
        "sgd": Stochastic gradient descent
        "adagrad": Adaptive gradient descent
    learning_rate: float
        Learning rate.
    validation_data: tuple, optional
        Dataset (X, y) used for validation.
    max_iter: int, optional
        Maximum number of iterations used in the optimizer.
    batch_size: int
        Size of the batch at each iteration.
    one_hot: boolen
        Whether label set is nominal or categorical.
    """
    def __init__(self, optimizer="adagrad", learning_rate=0.01, validation_data=None, max_iter=1000, batch_size=32, one_hot=False):
        self.model = NeuralNetwork(optimizer=optimizer, learning_rate=learning_rate, loss="cross_entropy", layers=None, validation_data=validation_data, max_iter=max_iter, batch_size=batch_size, one_hot=one_hot)

    def fit(self, X, y):
        """Fit the model to data matrix X and targets y."""
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        self.model.add(Dense(n_classes, input_shape=(n_features, )))
        self.model.add(Activation("softmax"))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
