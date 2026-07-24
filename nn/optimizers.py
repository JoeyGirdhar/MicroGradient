"""
Optimizers: given the parameters and their gradients, take one step downhill.

All three update parameters *in place* (param -= ...), so the network keeps
using the same array objects it was built with.

    SGD       plain gradient descent
    Momentum  SGD + velocity, rolls through small bumps and speeds up
    Adam      per-parameter adaptive learning rates; the sensible default
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        for v, p, g in zip(self.velocities, params, grads):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # 1st moment (mean of gradients)
        self.v = None  # 2nd moment (mean of squared gradients)
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            # Bias-correct the moment estimates (they start at zero).
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
