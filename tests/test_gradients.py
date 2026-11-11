"""Numerical gradient checks for the backward passes.

Run with pytest, or directly: python tests/test_gradients.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nn import Sequential, Dense, ReLU, Tanh, SoftmaxCrossEntropy, MSE


def _relative_error(a, b):
    return np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b))


def _numeric_param_grads(model, loss_fn, x, y, eps=1e-5):
    """Central-difference gradient of the loss w.r.t. every model parameter."""
    numeric = []
    for p in model.params():
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            original = p[idx]

            p[idx] = original + eps
            loss_plus = loss_fn.forward(model.forward(x), y)

            p[idx] = original - eps
            loss_minus = loss_fn.forward(model.forward(x), y)

            p[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        numeric.append(g)
    return numeric


def _check(model, loss_fn, x, y, tol=1e-6):
    # Analytic gradients via forward + backward.
    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())
    analytic = model.grads()

    numeric = _numeric_param_grads(model, loss_fn, x, y)

    worst = 0.0
    for a, n in zip(analytic, numeric):
        worst = max(worst, _relative_error(a, n).max())
    assert worst < tol, f"gradient check failed: max relative error {worst:.2e}"
    return worst


def test_classification_gradients():
    """Dense -> ReLU -> Dense with softmax cross-entropy."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 4))
    y = rng.integers(0, 3, size=8)
    model = Sequential([Dense(4, 5, seed=1), ReLU(), Dense(5, 3, seed=2)])
    worst = _check(model, SoftmaxCrossEntropy(), x, y)
    print(f"[classification] max relative error: {worst:.2e}  OK")


def test_regression_gradients():
    """Dense -> Tanh -> Dense with mean squared error."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3))
    y = rng.standard_normal((8, 1))
    model = Sequential([Dense(3, 6, seed=1), Tanh(), Dense(6, 1, seed=2)])
    worst = _check(model, MSE(), x, y)
    print(f"[regression]     max relative error: {worst:.2e}  OK")


if __name__ == "__main__":
    test_classification_gradients()
    test_regression_gradients()
    print("\nAll gradient checks passed. Backprop is verified correct.")
