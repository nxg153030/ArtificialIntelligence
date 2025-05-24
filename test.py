def hello(b):
    b.append(1)
    print(b)


if __name__ == '__main__':
    a = [1, 2, 3]
    hello(a)
    print(a)