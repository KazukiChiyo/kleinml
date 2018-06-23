"""
Author: Kexuan Zou
Date: Jun 22, 2018
Reference: https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
"""

import numpy as np
import sys
sys.path.append("../neural_network")
from funcs import Sigmoid, Softmax, ReLU, TanH
from copy import copy

class Layer(object):
    """Layer base class.
    """
    # set the shape of the inbound nodes
    def set_input_shape(self, shape):
        self.input_shape = shape

    # number of trainables in the layer
    def n_parameters(self):
        return 0;

    # forward signal to next outbound nodes
    def forward(self, X, training):
        raise NotImplementedError

    # receives the accumated gradient w.r.t. to output and calculates the gradient w.r.t. to the output of the previous layer
    def backward(self, accum_grad):
        raise NotImplementedError

    # outbound nodes shape
    def output_shape(self):
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer class that implements the operation: outputs = inputs*w + b.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The input shape of the first layer.
    """
    def __init__(self, n_units, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True

    def initialize(self, optimizer):
        self.w = np.random.uniform(0, 1, (self.input_shape[0], self.n_units))
        self.b = np.zeros((1, self.n_units))
        self.w_opt, self.b_opt = copy(optimizer), copy(optimizer)

    def n_parameters(self):
        return self.w.shape[0]*self.w.shape[1] + self.b.shape[1]

    def forward(self, X, training=True):
        self.layer_input = X
        return X.dot(self.w) + self.b

    def backward(self, accum_grad):
        w_copy = self.w
        if self.trainable:
            g_w = self.layer_input.T.dot(accum_grad) # gradient w.r.t. w
            g_b = np.sum(accum_grad) # gradient w.r.t. b
            self.w = self.w_opt.update(self.w, g_w)
            self.b = self.b_opt.update(self.b, g_b)
        return accum_grad.dot(w_copy.T) # gradient w.r.t. to the output of the previous layer

    def output_shape(self):
        return (self.n_units, )

activation_functions = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
    "tanh": TanH,
}

class Activation(Layer):
    """Activation layer class that provides different types of nonlinearities for use in neural networks: outputs = activation(inputs).
    Parameters:
    -----------
    name: string
        name of the activation function to use.
    """
    def __init__(self, name):
        self.activation = activation_functions[name]()
        self.trainable = True

    def forward(self, X, training=True):
        self.layer_input = X
        return self.activation(X)

    def backward(self, accum_grad):
        return accum_grad*self.activation.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape
