import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    f = sigmoid(x)
    return f*(1 - f)


if __name__ == "__main__":
    x = np.arange(-2, 1, 0.5)
    print("X =", x)
    print("Sigmoid(x) =",sigmoid(x))
    print("Derivative =", derivative_sigmoid(x))