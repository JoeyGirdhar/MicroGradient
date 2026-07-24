"""
Sequential: a thin container that chains layers together.

forward()  runs the layers front-to-back.
backward() runs them back-to-front, threading the gradient through each one.
params()/grads() flatten every layer's parameters so an optimizer can update
them in one loop.

That's genuinely all a feed-forward network is: a list of layers plus the
discipline to walk it forwards, then backwards.
"""


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        # Walk layers in reverse; each returns the gradient for the layer before it.
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]

    # Convenience so `model(x)` works like `model.forward(x)`.
    def __call__(self, x):
        return self.forward(x)
