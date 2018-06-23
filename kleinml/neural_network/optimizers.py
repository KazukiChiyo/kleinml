"""
Author: Kexuan Zou
Date: June 22, 2018s
"""

import numpy as np

class SGD():
    """SGD optimizer.
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
