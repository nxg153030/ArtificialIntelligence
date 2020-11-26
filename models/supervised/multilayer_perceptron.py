import numpy as np
from common.activation_functions import sigmoid, sigmoid_prime

# XOR dataset for testing
X = [(0, 0, -1), (0, 1, -1), (1, 0, -1), (1, 1, -1)]
Y = [0, 1, 1, 0]


class MultiLayerPerceptron:
    def __init__(self, dimensions, x_train, y_train, hidden_layers=0, learning_rate=0.5, epochs=10):
        """
        :param dimensions: This should be equal to the number of columns for each training sample
        :param layers:
        :param learning_rate:
        """
        self.train_data = np.array(x_train)
        self.targets = np.array(y_train)
        self.weights = np.zeros((dimensions, dimensions))  # can add bias parameter here if needed
        self.num_hidden_layers = hidden_layers
        self.num_layers = self.num_hidden_layers + 2
        self.hidden_layers = np.zeros((hidden_layers, hidden_layers))
        self.alpha = learning_rate
        self.epochs = epochs

    def __repr__(self):
        pass

    def forward_pass(self, train_row):
        outputs = []
        for idx in range(0, self.num_layers):
            if idx == 0:
                output = np.dot(self.weights[idx], train_row)
                transformed_output = sigmoid(output)
                outputs.append(transformed_output)
            else:
                output = sum(weight * outputs[-1] for weight in self.weights[idx])
                transformed_output = sigmoid(output)
                outputs.append(transformed_output)

        return outputs[-1]

    def backprop(self):
        pass

    def update_weights(self):
        pass

    def fit(self):
        deltas = []
        for epoch in np.arange(0, self.epochs):
            for row, target in zip(self.train_data, self.targets):
                transformed_outputs = self.forward_pass(row)
                # delta = target - transformed_outputs
                # deltas.append(delta)
                self.backprop()  # backpropagate all the deltas --> How?
                self.update_weights()

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
    nn = MultiLayerPerceptron(dimensions=3, x_train=X, y_train=Y, epochs=100)
    nn.fit()
    preds = nn.predict(X)
    print(f'Predictions: {preds}')
    accuracy = nn.accuracy(preds)
    print(f'Accuracy: {accuracy}')
