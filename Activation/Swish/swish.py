import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    f = sigmoid(x)
    return f*(1 - f)


def Swish(x):
    return x*sigmoid(x)


def derivative_Swish(x):
    f = sigmoid(x)
    return f - x*derivative_sigmoid(x)


if __name__ == "__main__":
    x = np.arange(-2, 1, 0.5)
    print("X =", x)
    print("Swish(x) =",Swish(x))
    print("Derivative =", derivative_Swish(x))