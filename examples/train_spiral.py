"""
Flagship demo: classify the spiral dataset and visualize the decision boundary.

Why the spiral? Because it is NOT linearly separable. A single linear layer
(logistic regression) can only draw straight boundaries and tops out around
~50% accuracy here. Add one hidden layer with a ReLU and the same training loop
bends the boundary into spirals and reaches ~99%. Seeing that jump is the
clearest one-picture explanation of "why depth / nonlinearity matters."

Run:  python examples/train_spiral.py
Output: prints training progress and saves assets/spiral_decision_boundary.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy, Adam
from nn.utils import accuracy
from data import make_spiral


def build_model():
    # 2 inputs (x, y coords) -> 64 -> 64 -> 3 class logits.
    return Sequential([
        Dense(2, 64, seed=1), ReLU(),
        Dense(64, 64, seed=2), ReLU(),
        Dense(64, 3, seed=3),
    ])


def train(model, x, y, epochs=1000, lr=0.05, verbose=True):
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(lr=lr)
    for epoch in range(1, epochs + 1):
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        opt.step(model.params(), model.grads())
        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {accuracy(logits, y):.3f}")
    return model


def plot_decision_boundary(model, x, y, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: save to file without a display
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. `pip install matplotlib` to enable.")
        return

    pad = 0.3
    x_min, x_max = x[:, 0].min() - pad, x[:, 0].max() + pad
    y_min, y_max = x[:, 1].min() - pad, x[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.forward(grid).argmax(axis=1).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap="brg")
    plt.scatter(x[:, 0], x[:, 1], c=y, s=18, cmap="brg", edgecolors="k", linewidths=0.3)
    plt.title("From-scratch neural net: learned spiral decision boundary")
    plt.xlabel("x1"); plt.ylabel("x2")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved decision-boundary plot to {out_path}")


def main():
    x, y = make_spiral(points_per_class=150, num_classes=3, seed=0)
    model = build_model()
    train(model, x, y, epochs=1000, lr=0.05)
    final_acc = accuracy(model.forward(x), y)
    print(f"\nFinal training accuracy: {final_acc:.3f}")
    out = os.path.join(os.path.dirname(__file__), "..", "assets", "spiral_decision_boundary.png")
    plot_decision_boundary(model, x, y, os.path.abspath(out))


if __name__ == "__main__":
    main()
