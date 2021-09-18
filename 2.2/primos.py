from sys import argv as args
from primelib import primos


if __name__ == "__main__":
    if len(args) != 2:
        print("Invalid number of arguments\nInput the first n primes only")
    else:
        n = int(args[1])
        primelist = primos(n)
        for prime in primelist:
            print(prime)
