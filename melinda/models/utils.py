from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


def logit_transform_inverse(data, datamin, datamax):
    dataout = (datamin + datamax * np.exp(data)) / (1 + np.exp(data))
    return dataout


def quick_logit(x, x_min, x_max, eps):
    x_norm = (x - x_min) / (x_max - x_min)
    logit = np.log(x_norm / (1 - x_norm))
    return logit


class LogitScaler(TransformerMixin, BaseEstimator):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.x_max = None
        self.x_min = None

    def fit(self, X, y=None):
        self.x_min = np.min(X, axis=0) - self.eps
        self.x_max = np.max(X, axis=0) + self.eps
        return self

    def transform(self, X):
        return quick_logit(X, self.x_min, self.x_max, self.eps)

    def inverse_transform(self, X):
        return logit_transform_inverse(X, self.x_min, self.x_max)