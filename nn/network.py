"""Sequential container."""

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

    def __call__(self, x):
        return self.forward(x)
