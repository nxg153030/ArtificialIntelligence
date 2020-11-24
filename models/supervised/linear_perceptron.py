import numpy as np
from common.activation_functions import sigmoid, sigmoid_prime

# XOR dataset for testing
X = [(0, 0, -1), (0, 1, -1), (1, 0, -1), (1, 1, -1)]
Y = [0, 1, 1, 1]


class LinearPerceptron:
    def __init__(self, dimensions, X_train, Y_train, layers=None, learning_rate=0.5, epochs=10):
        """
        :param dimensions: This should be equal to the number of columns for each training sample
        :param layers:
        :param learning_rate:
        """
        self.train_data = np.array(X_train)
        self.targets = np.array(Y_train)
        self.weights = np.zeros(dimensions)
        self.layers = layers
        self.alpha = learning_rate
        self.epochs = epochs

    def __repr__(self):
        pass

    def forward_pass(self):
        pass

    def backprop(self):
        pass

    def update_weights(self, train_row, target):
        output = np.dot(self.weights, train_row)
        delta = target - sigmoid(output)
        self.weights = self.weights + (self.alpha * delta * train_row * sigmoid_prime(output))

    def fit(self):
        for epoch in np.arange(0, self.epochs):
            for row, target in zip(self.train_data, self.targets):
                self.update_weights(row, target)

    def predict(self, test_data):
        predictions = []
        for i in range(len(test_data)):
            output = np.dot(self.weights, test_data[i])
            activated_output = sigmoid(output)
            if activated_output >= 0.5:
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)
        return predictions

    def accuracy(self, predictions):
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == self.targets[i]:
                correct += 1
        return correct/len(predictions)


if __name__ == '__main__':
    nn = LinearPerceptron(dimensions=3, X_train=X, Y_train=Y, epochs=100)
    nn.fit()
    preds = nn.predict(X)
    print(f'Predictions: {preds}')
    accuracy = nn.accuracy(preds)
    print(f'Accuracy: {accuracy}')
