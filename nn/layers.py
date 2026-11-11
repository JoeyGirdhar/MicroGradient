"""Fully connected layer."""

import numpy as np


class Dense:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # He init
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        # grad is dL/dz with shape (batch, out_features)
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]
