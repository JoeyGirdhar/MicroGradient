"""
Optional benchmark: handwritten-digit classification on MNIST.

This is the recognizable "does it actually work on real data?" test. The same
library, scaled up to 784 inputs and 10 classes, reaches ~97-98% test accuracy
with a couple of hidden layers and mini-batch training -- competitive with what
you'd get from a framework, but every line is in this repo.

First run downloads MNIST (~11 MB, cached in data/mnist/). Needs a network
connection once; fully offline afterwards. Training is CPU-only and takes a
couple of minutes.

Run:  python examples/train_mnist.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy, Adam
from nn.utils import accuracy, iterate_minibatches
from data import load_mnist


def main():
    print("Loading MNIST (downloads once, then cached)...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"train: {x_train.shape}, test: {x_test.shape}")

    model = Sequential([
        Dense(784, 256, seed=1), ReLU(),
        Dense(256, 128, seed=2), ReLU(),
        Dense(128, 10, seed=3),
    ])
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(lr=0.001)

    epochs = 10
    batch_size = 128
    for epoch in range(1, epochs + 1):
        start = time.time()
        running = 0.0
        n_batches = 0
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, seed=epoch):
            logits = model.forward(xb)
            running += loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            opt.step(model.params(), model.grads())
            n_batches += 1
        test_acc = accuracy(model.forward(x_test), y_test)
        print(f"epoch {epoch:2d}  loss {running / n_batches:.4f}  "
              f"test_acc {test_acc:.4f}  ({time.time() - start:.1f}s)")

    print(f"\nFinal test accuracy: {accuracy(model.forward(x_test), y_test):.4f}")


if __name__ == "__main__":
    main()
