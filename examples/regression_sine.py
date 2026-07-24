"""
Regression demo: fit y = sin(x) with the same library, swapping the loss.

Shows the toolkit isn't classification-only. Same layers, but a Tanh network and
mean squared error let it trace a smooth curve. Saves a plot of the fit.

Run:  python examples/regression_sine.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nn import Sequential, Dense, Tanh, MSE, Adam


def main():
    rng = np.random.default_rng(0)
    x = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
    y = np.sin(x) + rng.normal(0, 0.05, x.shape)

    model = Sequential([
        Dense(1, 64, seed=1), Tanh(),
        Dense(64, 64, seed=2), Tanh(),
        Dense(64, 1, seed=3),
    ])
    loss_fn = MSE()
    opt = Adam(lr=0.01)

    for epoch in range(1, 3001):
        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        model.backward(loss_fn.backward())
        opt.step(model.params(), model.grads())
        if epoch % 500 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  mse {loss:.5f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, s=8, alpha=0.4, label="noisy data")
    plt.plot(x, model.forward(x), color="crimson", linewidth=2, label="network fit")
    plt.legend(); plt.title("From-scratch net fitting sin(x)")
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "sine_fit.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved fit plot to {out}")


if __name__ == "__main__":
    main()
