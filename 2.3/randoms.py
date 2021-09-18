from numpy.random import uniform as unif
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def uniforme() -> None:
    dist = unif(0, 1, 1000)
    print(dist)
    plt.hist(dist, range=(0.0, 1.0))
    plt.xlabel("Value of generated random numbers")
    plt.ylabel("Number of occurances inside the sample")
    plt.title("Histogram of a random uniform distribution")
    plt.show()


def randombins(n: int) -> Tuple[np.ndarray]:
    random1 = np.random.randint(2, size=n)
    random2 = np.random.randint(2, size=n)
    return random1, random2


if __name__ == "__main__":
    uniforme()
    r31, r32 = randombins(3)
    print("Sequences of random binary numbers (3): ")
    print("Sequence 1: ", r31)
    print("Sequence 2: ", r32)
    r3dot = np.dot(r31, r32)
    print("Dot product: ", r3dot)
    r1001, r1002 = randombins(100)
    print("Sequences of random binary numbers (100): ")
    print("Sequence 1: ", r1001)
    print("Sequence 2: ", r1002)
    r100dot = np.dot(r1001, r1002)
    print("Dot product: ", r100dot)
    r10001, r10002 = randombins(1000)
    print("Sequences of random binary numbers (1000): ")
    print("Sequence 1: ", r1001)
    print("Sequence 2: ", r1002)
    r1000dot = np.dot(r10001, r10002)
    print("Dot product: ", r1000dot)
    #  The dot product is roughly 1/4 the size of the sample
    #  This is because the probability of two separate variables equaling 1 is 1/4
    #  1/2 * 1/2 = 1/4
    #  The dot product will simply sum all of the 1s that are in the same position of the lists
    #  Therefore, the result is approximatedly 1/4 of the original size
