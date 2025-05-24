import numpy as np
from common.activation_functions import sigmoid, sigmoid_prime

# XOR dataset for testing
X = [(0, 0, -1), (0, 1, -1), (1, 0, -1), (1, 1, -1)]
Y = [0, 1, 1, 0]

def transfer_derivative(output):
    return output * (1.0 - output)


class MultiLayerPerceptron:
    def __init__(self, dimensions, x_train, y_train, hidden_layer_size, hidden_layers=0, learning_rate=0.5, epochs=10,
                 activation_func=sigmoid):
        """
        :param dimensions: This should be equal to the number of columns for each training sample
        :param layers:
        :param learning_rate:
        References: https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html
        Should no. of inputs != no. of hidden layers?
        """
        self.train_data = np.array(x_train)
        self.targets = np.array(y_train)
        self.input_layer_weights = []
        self.hidden_layer_weights = []
        self.output_layer_weights = []
        self.weights = []
        self.input_layer_size = len(self.train_data[0])
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = len(self.targets[0])
        self.num_hidden_layers = hidden_layers
        self.num_layers = self.num_hidden_layers + 2
        self.hidden_layers = np.zeros((hidden_layers, hidden_layers))
        self.alpha = learning_rate
        self.epochs = epochs
        self.activation_func = activation_func

    def init_weights(self):
        """
        We need 3 separate weight matrices
        One for the input layer -> first hidden layer
        One for hidden layer -> hidden layer
        One for hidden layer -> output layer
        TODO: Modify this so that there can be N hidden layers (N weight matrices)
        """
        self.input_layer_weights = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.hidden_layer_weights = np.random.randn(self.hidden_layer_size, self.hidden_layer_size)
        self.output_layer_weights = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        self.weights.append(self.input_layer_weights)
        self.weights.append(self.hidden_layer_weights)
        self.weights.append(self.output_layer_weights)

    def forward_pass(self, train_row):
        """
        :param train_row:
        :return:
        """
        outputs, deltas = [], []
        for idx in range(self.num_layers):
            if idx == 0:
                activations = train_row
                outputs.append(train_row)
            else:
                activations = np.dot(outputs[-1], self.weights[idx]) # think this has to be scalar
                transformed_output = self.activation_func(activations)


        # for idx in range(0, self.num_layers):
        #     if idx == 0:
        #         output = np.dot(self.weights[idx], train_row)
        #         transformed_output = sigmoid(output)
        #         outputs.append(transformed_output)
        #     else:
        #         output = sum(weight * outputs[-1] for weight in self.weights[idx])
        #         transformed_output = sigmoid(output)
        #         outputs.append(transformed_output)

        # compute deltas for all layers
        # for idx in range(self.num_layers):
        #     delta = None

        return outputs[-1]

    def backprop(self, output, target):
        """
        This is incomplete
        """
        delta = output - target
        error = delta * sigmoid_prime(output)

    def update_weights(self):
        pass

    def fit(self):
        deltas = []
        self.init_weights()
        for epoch in np.arange(0, self.epochs):
            for row, target in zip(self.train_data, self.targets):
                transformed_output = self.forward_pass(row)
                # delta = target - transformed_outputs
                # deltas.append(delta)
                # outputs.append(transformed_output)
                self.backprop(transformed_output, target)  # backpropagate all the deltas --> How?
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
