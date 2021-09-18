# Este programa despliega una serie de tiempo a partir de una base de datos
import pandas as pd
from pandas import Series
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import copy

from datetime import datetime

# datos tomados de https://finance.yahoo.com/
data = pd.read_csv('ETH-USD.csv', usecols=["Date", "Close"])
y = pd.read_csv('ETH-USD.csv')
print(data)
ym = np.mean(y)
ys = np.std(y)
print('Media:', ym)
print('Desv Estandar:', ys)
con = data['Date']
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# check datatype of index
var = data.index

data.plot()

plt.show()

# Ahora una definicion arbitraria de 'recesion': un
# dia seguido de al menos 3 dias seguidos a la baja es bit 1
# bit 0 en caso contrario

print(len(data))
# y = [0]*len(data)
yd = data.values.tolist()
# print(yd)
y = yd[0:len(data)]

for i in range(len(y) - 3):
    if y[i + 1] < y[i]:
        if y[i + 2] < y[i + 1]:
            if y[i + 3] < y[i + 2]:
                y[i] = 1
            else:
                y[i] = 0
        else:
            y[i] = 0
    else:
        y[i] = 0

print(y)
print(i)
lim = len(y) - 3
print(lim)

# plt.plot(y[0:lim])
# plt.show()

# Para fines comparativos, veamos
# una serie de tiempo de digitos binarios
# aleatorios e independientes

yrand = copy.copy(yd[0:len(data)])  # ¿porque no hacemos yrand = yd?

import random

for i in range(len(yrand) - 3):
    yrand[i] = random.randint(0, 1)

print(yrand)
lim = len(yrand) - 3

plt.plot(yrand[0:lim])
plt.show()


