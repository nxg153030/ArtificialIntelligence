from math import cos, pi
from math import exp


def rastrigin(x, n):
    """
    :param x: position vector
    :param n: no. of dimensions
    :return: scalar value representing the magnitude of your position
    Global minimum: 0
    """
    A = 10
    return round((A * n) + sum(x_i ** 2 - A * cos(2 * pi * x_i) for x_i in x), 4)


def sphere(x, n):
    return sum(x_i**2 for x_i in x)


def ackley(x, n):
    pass


def schwefel(x, n):
    pass
