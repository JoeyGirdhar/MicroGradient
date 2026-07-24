"""Small training helpers: accuracy, mini-batches, splits, standardization."""

import numpy as np


def accuracy(logits, y_int):
    """Fraction of rows whose argmax matches the integer label."""
    return (logits.argmax(axis=1) == y_int).mean()


def iterate_minibatches(x, y, batch_size, shuffle=True, seed=None):
    """Yield (x_batch, y_batch) tuples covering one epoch."""
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        yield x[batch], y[batch]


def train_test_split(x, y, test_frac=0.2, seed=0):
    n = x.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]


def standardize(x, mean=None, std=None):
    """Zero-mean, unit-variance per feature. Fit stats on train, reuse on test."""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0) + 1e-8
    return (x - mean) / std, mean, std
