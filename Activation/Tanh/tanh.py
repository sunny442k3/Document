import numpy as np


def Tanh(x):
    num = np.exp(x) - np.exp(-x)
    denom = np.exp(x) + np.exp(-x)
    return num/denom


def derivative_Tanh(x):
    return 1 - Tanh(x)


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    print("X =", x)
    print("Sigmoid(x) =",Tanh(x))
    print("Derivative =", derivative_Tanh(x))