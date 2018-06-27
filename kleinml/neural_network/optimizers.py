"""
Author: Kexuan Zou
Date: June 22, 2018s
"""

import numpy as np

class SGD(object):
    """SGD optimizer for training neural network.
    Parameters:
    -----------
    learning_rate: float
        Learning rate.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w_update = None

    def update(self, w, grad):
        if self.w_update is None:
            self.w_update = np.zeros_like(w)
        return w - self.learning_rate*grad

class Adagrad(object):
    """Adagrad optimizer for training neural network.
    Parameters:
    -----------
    learning_rate: float
        Learning rate.
    eta0: float, optional
        Smoothing term.
    """
    def __init__(self, learning_rate=0.01, eta0=1e-8):
        self.learning_rate = learning_rate
        self.G = None
        self.eta0 = eta0

    def update(self, w, grad):
        if self.G is None:
            self.G = np.zeros_like(w)
        self.G += np.square(grad)
        return w - self.learning_rate*grad/np.sqrt(self.G + self.eta0)
