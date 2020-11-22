from math import exp
import numpy as np


def sigmoid(x):
    return 1/(1 + exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    pass


def maximum_likelihood(inputs, targets, ):
    pass


def squared_error_loss(inputs, targets, weights, learning_rate):
    num_samples = len(inputs)
    loss = 0
    for i in range(num_samples):
        weighted_feature_vector = np.dot(weights, inputs[i])
        loss += (targets[i] - sigmoid(inputs) * weights) ** 2
        weights += learning_rate * loss * (inputs[i]) * sigmoid_prime(inputs[i] * sigmoid_prime(weighted_feature_vector))
    loss = loss / num_samples
    return loss


class LogisticRegression:
    def __init__(self, data, classes, iterations, epochs, batch_size, loss_function, learning_rate=0.001, bias=10.0, penalty='l1'):
        self.data = data
        self.classes = classes
        self.iterations = iterations
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.weights = [0 for _ in range(len(self.data[0]))]
        self.learning_rate = learning_rate
        self.bias = bias
        self.penalty = penalty

    def fit(self):
        for i in range(self.epochs):
            weighted_feature_vector = np.dot(self.weights, self.data[i])
            delta = 0
            self.weights = self.weights + (self.learning_rate * delta) * (sigmoid_prime(weighted_feature_vector) * self.data[i])

    def predict(self):
        pass




