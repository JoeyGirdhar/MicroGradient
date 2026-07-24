"""
A minimal neural-network library built from scratch on top of NumPy.

Public API:

    from nn import Sequential, Dense, ReLU, Tanh, Sigmoid
    from nn import SoftmaxCrossEntropy, MSE
    from nn import SGD, Momentum, Adam
"""

from .layers import Dense
from .activations import ReLU, Tanh, Sigmoid
from .losses import SoftmaxCrossEntropy, MSE
from .optimizers import SGD, Momentum, Adam
from .network import Sequential

__all__ = [
    "Sequential",
    "Dense",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "SoftmaxCrossEntropy",
    "MSE",
    "SGD",
    "Momentum",
    "Adam",
]
