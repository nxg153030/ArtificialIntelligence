import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return np.multiply(sig, 1 - sig, out=sig) # in-place multiplication