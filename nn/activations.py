"""Activation layers."""

import numpy as np


class ReLU:
    """f(x) = max(0, x). The workhorse nonlinearity for hidden layers."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        # d/dx relu(x) = 1 where x > 0, else 0.
        return grad * self.mask

    def params(self):
        return []

    def grads(self):
        return []


class Tanh:
    """f(x) = tanh(x). Smooth, zero-centered; nice for regression demos."""

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        # d/dx tanh(x) = 1 - tanh(x)^2
        return grad * (1.0 - self.out ** 2)

    def params(self):
        return []

    def grads(self):
        return []


class Sigmoid:
    """f(x) = 1 / (1 + e^-x). Squashes to (0, 1)."""

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        return grad * self.out * (1.0 - self.out)

    def params(self):
        return []

    def grads(self):
        return []
