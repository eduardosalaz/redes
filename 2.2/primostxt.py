from sys import argv as args
from primelib import primos


if __name__ == "__main__":
    if len(args) != 2:
        print("Invalid number of arguments\nInput the name of the txt file to be read")
    else:
        filename = str(args[1])
        bufferin = open(filename, "r")
        n = int(bufferin.read())
        bufferin.close()
        primelist = primos(n)
        with open("primes.txt", "w") as bufferout:
            for prime in primelist:
                bufferout.write(f"{prime}\n")
        print("Wrote to primes.txt")

