import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit_transform(self, x):
        # x: np 2d array(shape: N x d, where N is the number of data samples and d is the original dimension)
        cov_matrix = np.matmul(x.T, x)
        eig_values, eig_vectors = eigh(cov_matrix)

        self.w = eig_vectors[-self.n_components:].T
        return np.matmul(x, self.w)


if __name__ == '__main__':
    pca = PCA(n_components=2)
    x = np.random.normal(0, 1, size=(10, 3))
    x_reduced = pca.fit_transform(x)
    print(x_reduced)