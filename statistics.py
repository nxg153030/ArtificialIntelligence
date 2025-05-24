import math


def mean(dataset):
    return sum(dataset) / len(dataset)


def variance(dataset):
    numerator = 0
    _mean = mean(dataset)
    for num in dataset:
        numerator += (num - _mean) ** 2
    return numerator / len(dataset)


def variance_n_minus_one(dataset):
    numerator = 0
    _mean = mean(dataset)
    for num in dataset:
        numerator += (num - _mean) ** 2
    return numerator / (len(dataset) - 1)


def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)


def n_choose_k(n, k):
    denominator = factorial(k)
    numerator = 1
    for i in range(k):
        numerator *= (n - i)
    return numerator // denominator


if __name__ == '__main__':
    # d1 = [0, 0, 5, 5, 100]
    # d1 = [100] * 9 + [10000]
    # d1 = [0, 0, 0, 0, 10000]
    # print(f'Mean: {mean(d1)}')
    # print(f'Variance with original formula: {variance(d1)}')
    # print(f'Standard deviation with original formula: {math.sqrt(variance(d1))}')
    # print(f'Variance with n-1 formula: {variance_n_minus_one(d1)}')
    # print(f'Standard deviation with n-1 formula: {math.sqrt(variance_n_minus_one(d1))}')
    p_make = 0.3
    p_fail = 1 - p_make
    print(n_choose_k(6, 6))
    n = 6
    k = 2
    print(n_choose_k(n, k) * (0.3 ** k) * (0.7 ** (n - k)))

