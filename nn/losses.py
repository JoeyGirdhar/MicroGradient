"""
Loss functions. Each computes a scalar loss in forward() and, in backward(),
the gradient of that loss w.r.t. the network's *output* -- the seed gradient
that gets fed back through the layers.

SoftmaxCrossEntropy fuses the softmax and the cross-entropy on purpose: the
combined gradient simplifies to the famous, numerically-stable expression
(probs - one_hot) / N. Doing them separately is both slower and flakier.
"""

import numpy as np


class SoftmaxCrossEntropy:
    """For multi-class classification. Expects raw logits + integer labels."""

    def forward(self, logits, y_int):
        # Subtract the row max before exp() for numerical stability (no overflow).
        shift = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shift)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y = y_int
        n = logits.shape[0]
        # Negative log-likelihood of the correct class, averaged over the batch.
        correct_logprobs = -np.log(self.probs[np.arange(n), y_int] + 1e-12)
        return correct_logprobs.mean()

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1.0  # subtract 1 from the true-class prob
        grad /= n
        return grad


class MSE:
    """Mean squared error, for regression."""

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        # d/dpred mean((pred - target)^2) = 2 (pred - target) / N_elements
        return 2.0 * (self.pred - self.target) / self.pred.size
