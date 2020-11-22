"""
Implementation of the Linear Perceptron Algorithm for Neural Networks
"""
import pandas as pd
import numpy as np
import sys
from math import exp


def main():
    if len(sys.argv) != 5:
        print("ERROR! Incorrect number of arguments! Exiting Program . . .")
        exit()
    training_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    learning_rate = float(sys.argv[3])
    num_iterations = int(sys.argv[4])
    training_file = pd.read_table(training_file_name)
    test_file = pd.read_table(test_file_name)
    X_train = training_file.drop('class', axis=1)
    Y_train = training_file['class']
    X_test = test_file.drop('class', axis=1)
    Y_test = test_file['class']
    weightVec = fit(X_train, Y_train, num_iterations, learning_rate)
    predicted_Y_train = predict(weightVec, X_train)
    predicted_Y_test = predict(weightVec, X_test)
    print('Accuracy on training set (', len(X_train), 'instances): ', accuracy(predicted_Y_train, Y_train), '%')
    print('Accuracy on test set (', len(X_test), 'instances): ', accuracy(predicted_Y_test, Y_test), '%')


# classify all training instances return weight vector
# prints the weight values after each iteration
# X = training instances
# Y = output values for each training instance
# Returns the weight matrix
def fit(X, Y, numIter, learning_rate):
    weight_update = [0 for _ in range(len(X.keys()))]
    i, j = 0, 0
    while (i < numIter and j < numIter):
        if (i >= len(X)):
            i = 0
        wtDotX = np.dot(weight_update, X.iloc[i])
        delta = Y.iloc[i] - sigmoid(wtDotX)
        weight_update = weight_update + learning_rate * delta * (X.iloc[i]) * sigmoid_prime(wtDotX)
        weight_update = np.round(weight_update, decimals=4)
        print('After iteration', j + 1, ': ', end='')
        for k in range(0, len(X.keys())):
            print('w(', X.keys()[k], ')=', weight_update[k], ',', end='')
        updatedWtDotX = np.dot(weight_update, X.iloc[i])
        output = np.round(sigmoid(updatedWtDotX), 4)
        print('Output = ', output)
        i += 1
        j += 1
    return weight_update


def sigmoid(X):
    return 1 / (1 + exp(-X))


def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))


# return predicted Y values for test set
def predict(weightVec, test):
    predicted_Y_test = [None] * len(test)
    for i in range(0, len(test)):
        wtDotTest = np.dot(weightVec, test.iloc[i])
        output = np.round(sigmoid(wtDotTest), 4)
        if (output >= 0.5):
            predicted_Y_test[i] = 1
        else:
            predicted_Y_test[i] = 0
    return predicted_Y_test


def accuracy(predicted_Y_vec, Yvalues):
    correct_counter = 0
    for i in range(0, len(Yvalues)):
        if (predicted_Y_vec[i] == Yvalues.iloc[i]):
            correct_counter += 1
    accuracy = (correct_counter / len(predicted_Y_vec)) * 100
    accuracy = np.round(accuracy, 1)
    return accuracy


if __name__ == '__main__':
    main()
