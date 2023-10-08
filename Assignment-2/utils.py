import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (relu(x) > 0).astype(float)


def loss_func(y, y_pred):
    return 0.5 * (y - y_pred) ** 2
