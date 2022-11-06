import numpy as np


def Leaky_ReLU(x, alpha):
    return np.where(x >= 0, x, alpha*x)

def derivative_Leaky_ReLU(x, alpha):
    return np.where(x >= 0, 1, alpha)


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    print("X =", x)
    print("Leaky_ReLU(x) =",Leaky_ReLU(x))
    print("Derivative =", derivative_Leaky_ReLU(x))