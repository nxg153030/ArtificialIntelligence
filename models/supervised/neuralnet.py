import numpy as np
from common.activation_functions import *


class NeuralNet:
    def __init__(self, layers=[], learning_rate=0.1):
        self.data = []
        self.weights = []
        self.layers = layers
        self.alpha = learning_rate

    def __repr__(self):
        pass

    def initialize_weights(self):
        for i in np.arange(0, len(self.layers) - 2):
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1)  # +1 for the bias term
            # scale w by dividing the square root of the no. of nodes in the current layer,
            # normalizing the variance of each neuron's output
            self.weights.append(w / np.sqrt(self.layers[i]))

        # last 2 layers are a special case where the input connections
        # need a bias term but the output does not
        w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        self.weights.append(w / np.sqrt(self.layers[-2]))

    def update_weights(self, train_row, target):
        A = [np.atleast_2d(train_row)]
        for layer in np.arange(0, len(self.weights)):
            net = A[layer].dot(self.weights[layer])
            out = sigmoid(net)
            A.append(out)

        error = A[-1] - y
        D = [error * sigmoid]

    def fit(self, train_data, target, epochs=10):
        # insert a column of 1s to treat bias as trainable parameter
        train_data = np.c_[train_data, np.ones((train_data.shape[0]))]

        for epoch in np.arange(0, epochs):
            for row, target in zip(train_data, target):
                self.update_weights(row, target)

