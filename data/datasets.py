"""
Datasets.

`make_spiral` and `make_moons` are generated in pure NumPy -- no downloads, no
dependencies -- so the main demos run instantly and offline. They are also
*deliberately* not linearly separable, which is the whole point: a linear model
fails on them, and a small neural net succeeds. That contrast is the lesson.

`load_mnist` is optional and downloads the classic handwritten-digit dataset the
first time it runs (cached afterwards). Only the MNIST demo needs it.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np


def make_spiral(points_per_class=100, num_classes=3, noise=0.2, seed=0):
    """The classic spiral toy dataset (as in Stanford's CS231n).

    Returns X of shape (points_per_class * num_classes, 2) and integer labels y.
    """
    rng = np.random.default_rng(seed)
    n, k = points_per_class, num_classes
    x = np.zeros((n * k, 2))
    y = np.zeros(n * k, dtype=int)
    for class_idx in range(k):
        ix = range(n * class_idx, n * (class_idx + 1))
        radius = np.linspace(0.0, 1.0, n)
        theta = np.linspace(class_idx * 4, (class_idx + 1) * 4, n) + rng.normal(0, noise, n)
        x[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[ix] = class_idx
    return x, y


def make_moons(n_samples=400, noise=0.15, seed=0):
    """Two interleaving half-moons -- a 2-class version of the same idea."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    t = np.linspace(0, np.pi, n)
    outer = np.c_[np.cos(t), np.sin(t)]
    inner = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    x = np.vstack([outer, inner]) + rng.normal(0, noise, (2 * n, 2))
    y = np.array([0] * n + [1] * n)
    return x, y


# ---------------------------------------------------------------------------
# Optional: MNIST (only imported/used by examples/train_mnist.py)
# ---------------------------------------------------------------------------

_MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols).astype(np.float64) / 255.0


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        _, _ = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(int)


def load_mnist(cache_dir=None):
    """Download (once) and load MNIST. Returns (x_train, y_train, x_test, y_test).

    Images are flattened to length-784 vectors and scaled to [0, 1].
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "mnist")
    os.makedirs(cache_dir, exist_ok=True)

    paths = {}
    for key, fname in _MNIST_FILES.items():
        dest = os.path.join(cache_dir, fname)
        _download(_MNIST_BASE + fname, dest)
        paths[key] = dest

    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])
    return x_train, y_train, x_test, y_test
