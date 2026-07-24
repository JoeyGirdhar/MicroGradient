"""
Entry point. Runs the spiral classification demo end-to-end:
trains a from-scratch neural net and saves a decision-boundary plot.

    python main.py

For the other demos, see the examples/ folder:
    python examples/regression_sine.py   # regression
    python examples/train_mnist.py        # MNIST benchmark (downloads data once)
    python tests/test_gradients.py        # verify backprop is correct
"""

from examples.train_spiral import main

if __name__ == "__main__":
    main()
