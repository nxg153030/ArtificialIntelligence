from common.activation_functions import (sigmoid, sigmoid_prime)
import numpy as np


# Mean Squared Error loss and its derivative
def mse_loss(y_true: np.ndarry, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size

# Simple neural network with one hidden layer
class ArtificialNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Initialize weights and biases
        # Each input neuron is connected to every hidden neuron.
        # The weight matrix must have one row for each input neuron and one column for each hidden neuron
        # If there are 3 input neurons and 4 hidden neurons, the weight matrix will be of shape (3, 4)
        # Each hidden neuron has a bias term. The bias vector must have one value for each hidden neuron
        # If there are 4 hidden neurons, the bias vector will be of shape (1, 4)
        # Each hidden neuron is connected to every output neuron.
        # The weight matrix must have one row for each hidden neuron and one column for each output neuron
        # If there are 4 hidden neurons and 2 output neurons, the weight matrix will be of shape (4, 2)
        # Each output neuron has a bias term. The bias vector must have one value for each output neuron
        # If there are 2 output neurons, the bias vector will be of shape (1, 2)
        # The weights and biases are initialized randomly and initialized with a normal distribution
        # The weights are initialized with a normal distribution with mean 0 and standard deviation 1
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output
    
    def backward(self, x: np.ndarray, y_true: np.ndarray, learning_rate):
        # Compute loss derivative
        loss_gradient = mse_loss_prime(y_true, self.final_output)

        # Backpropagation for output layer
        output_gradient = loss_gradient * sigmoid_prime(self.final_input)
        weights_hidden_output_gradient = np.dot(self.hidden_output.T, output_gradient)
        bias_output_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= learning_rate * weights_hidden_output_gradient
        self.bias_output -= learning_rate * bias_output_gradient
        self.weights_input_hidden -= learning_rate * weights_input_hidden_gradient
        self.bias_hidden -= learning_rate * bias_hidden_gradient
    
    def train():
        pass

