import numpy as np


def ReLU(x):
    return np.where(x >= 0, x, 0)

def derivative_ReLU(x):
    return np.where(x >= 0, 1, 0)


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    print("X =", x)
    print("ReLU(x) =",ReLU(x))
    print("Derivative =", derivative_ReLU(x))