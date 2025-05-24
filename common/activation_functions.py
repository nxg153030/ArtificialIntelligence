import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return np.multiply(sig, 1 - sig, out=sig) # in-place multiplication

def reLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def reLU_prime(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0) # returns 1 for positive x, 0 for non-positive x

def leaky_reLU(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x) # returns x for positive x, alpha * x for non-positive x

def leaky_reLU_prime(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1, alpha)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2 # derivative of tanh is 1 - tanh^2

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # For numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True) # Normalize to get probabilities

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_prime(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, 1, alpha * np.exp(x))

def swish(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x) # Swish activation function

def swish_prime(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig) # derivative of swish is sigmoid + x * sigmoid * (1 - sigmoid)