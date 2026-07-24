"""
Activation functions, implemented as layers.

Each activation is a tiny "layer" with a forward and a backward pass, so it
plugs into the same Sequential container as the Dense layer. That symmetry is
the whole trick behind backprop: every piece only needs to know how to
(1) transform its input going forward, and (2) turn a gradient of the loss
w.r.t. its *output* into a gradient w.r.t. its *input*.

None of these have trainable parameters, so params()/grads() return empty lists.
"""

import numpy as np


class ReLU:
    """f(x) = max(0, x). The workhorse nonlinearity for hidden layers."""

    def forward(self, x):
        # Cache which entries were positive; the gradient only flows through those.
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
