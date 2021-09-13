import numpy as np
import pandas as pd
from numpy.linalg import inv, pinv, LinAlgError
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, x, y):
        # x: np 2d array(shape: N x d, where N is the number of data samples and d is the original dimension)
        # y: np 1d array shape(shape: N, should be matched with x's shape)
        self.feat_dim = x.shape[1]
        assert x.shape[0] == y.shape[0]

        if self.fit_intercept:
            intercept = np.ones((x.shape[0], 1))
            x = np.concatenate([intercept, x], axis=1)

        xtx = np.matmul(x.T, x)
        try:
            xtx_inv = inv(xtx)
        except LinAlgError:
            xts_inv = pinv(x)
        self.w = np.matmul(np.matmul(xtx_inv, x.T), y.reshape(-1, 1))
        return np.matmul(x, self.w).squeeze()
        
    def predict(self, x):
        # x: 2d array(shape: N x d, where N is the number of data samples and d is the original dimension)
        assert self.feat_dim == x.shape[1]

        if self.fit_intercept:
            intercept = np.ones((x.shape[0], 1))
            x = np.concatenate([intercept, x], axis=1)
        return np.matmul(x, self.w).squeeze()

    def get_params(self):
        return self.w


if __name__ == '__main__':
    train_x = np.random.normal(0, 1, size=(10, 5))
    train_y = np.random.normal(0, 1, size=10)

    linear_regression = LinearRegression()
    linear_regression.fit(train_x, train_y)

    test_x = np.random.normal(0, 1, size=(2, 5))
    pred_y = linear_regression.predict(test_x)
    print(pred_y)