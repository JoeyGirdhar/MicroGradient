"""
The Dense (fully-connected) layer -- the only layer here with learnable weights.

Forward:   z = x @ W + b
Backward:  given dL/dz (the gradient of the loss w.r.t. this layer's output),
           compute dL/dW, dL/db, and dL/dx.

The three backward formulas are the entire "learning" mechanism of a
feed-forward net. They come straight from the chain rule:

    dL/dW = x^T @ (dL/dz)      # how each weight nudges the loss
    dL/db = sum over batch of dL/dz
    dL/dx = (dL/dz) @ W^T      # gradient handed back to the previous layer
"""

import numpy as np


class Dense:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization: keeps activation variance stable through ReLU nets.
        # Scaling by sqrt(2 / fan_in) is what lets deep-ish nets train at all.
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x  # cache input for the backward pass
        return x @ self.W + self.b

    def backward(self, grad):
        # grad is dL/dz with shape (batch, out_features)
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T  # dL/dx, passed to the layer before this one

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]
