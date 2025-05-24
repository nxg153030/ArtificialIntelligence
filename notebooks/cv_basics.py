import numpy as np


def convolution(input_matrix, conv_filter, stride=1, padding='valid'):
    """
    Slide the conv_filter over input matrix
    and do element-wise multiplication
    """
    output_height = output_width = len(input_matrix) - len(conv_filter) + 1
    output = np.zeros((output_height, output_width))
    for i in range(0, output_height):
        for j in range(0, output_width):
            temp_output = input_matrix[i:i + len(conv_filter), j:j + len(conv_filter)] * conv_filter
            output[i][j] = np.sum(temp_output)

    return output


if __name__ == '__main__':
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    conv_filter = np.array([[1, 1], [1, 1]])
    print(convolution(A, conv_filter))