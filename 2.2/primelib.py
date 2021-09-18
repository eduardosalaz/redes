from typing import List


def primos(n: int) -> List[int]:
    counter = 0
    i = 2
    primes = []
    while counter < n:
        flag = False
        for j in range(2, i // 2 + 1):
            if i % j == 0:
                flag = True
                break
        if not flag:
            primes.append(i)
            counter = counter + 1
        i = i + 1
    return primes
