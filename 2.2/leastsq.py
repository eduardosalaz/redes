from pandas import read_csv as csv
from sys import argv as args
import matplotlib.pyplot as plt
import numpy as np


def lstsq(X: np.ndarray, Y: np.ndarray) -> None:
    plt.scatter(X, Y)
    plt.title(f"Scatter plot of {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    Xmean = np.mean(X)
    Ymean = np.mean(Y)
    upper = 0
    lower = 0
    for i in range(len(X)):
        upper = upper + (X[i] - Xmean) * (Y[i] - Ymean)
        lower = lower + np.power((X[i] - Xmean), 2)
    weight = upper / lower
    epsilon = Ymean - weight * Xmean
    print(f"Weight of {weight}\nEpsilon of {epsilon}")
    Ypred = weight * X + epsilon  # wX + E
    plt.scatter(X, Y)
    plt.title("Least squares regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot([min(X), max(X)], [min(Ypred), max(Ypred)], color="r", label="Prediction")
    plt.show()


if __name__ == "__main__":
    if len(args) != 2:
        print("Invalid number of arguments\nInput the name of the csv file to be read")
    else:
        filename = str(args[1])
        data = csv(filename)
        x = data.x.values
        y = data.y.values
        lstsq(x, y)
