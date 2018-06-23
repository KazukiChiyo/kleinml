"""
Author: Kexuan Zou
Date: Jun 14, 2018
"""

import numpy as np
import numpy.linalg as la

class PCA(object):
    """Linear dimensionality reduction using eigendecomposition.
    Parameters:
    -----------
    n_components: int, or None
        Number of components to keep.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        pass

    def fit(self, X):
        X = np.array(X)
        if not self.n_components:
            self.n_components = np.min(X.shape[0], X.shape[1])
        cov_mat = np.cov(X.T)
        eigenval, eigenvec = la.eig(cov_mat)
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx][:self.n_components]
        self.components = np.atleast_1d(eigenvec[:, idx])[:, :self.n_components]
        return self

    def transform(self, X):
        return np.array(X).dot(self.components)
