#Este programa despliega visualmente un conjunto de datos en formato csv
import pandas
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt

filename=np.array(['.csv', '.csv', '.csv'])

i = 1
for f in filename:
    series = pandas.read_csv(f)
    series.plot()

print series
plt.show()

