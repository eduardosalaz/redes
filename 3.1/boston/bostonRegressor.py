from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from pandas import read_csv as csv
import numpy as np
from pickle import dump as save

if __name__ == '__main__':
    xNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    yNames = ['MEDV']

    xTrain = csv('xbostonTrain.csv', names=xNames)
    yTrain = np.ravel(csv('ybostonTrain.csv', names=yNames))  # 1d array expected not a column vector

    xTest = csv('xbostonTest.csv', names=xNames)
    yTest = csv('ybostonTest.csv', names=yNames)

    regressor = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(120, 120, 120, 120), max_fun=20000, max_iter=18000)
    # lbfgs works best for smaller datasets (<1000)
    # more iterations == better mse score but longer times
    # the current # of iterations is enough for lbfgs to converge, lower than 15,000 won't work
    regressor.fit(xTrain, yTrain)
    yTheoretical = regressor.predict(xTest)
    err = mse(np.ravel(yTest), yTheoretical)
    print(f'Error cuadrado medio sobre el conjunto de prueba: {err:.3f}')
    print(f'Raiz cuadrada del error: {np.sqrt(err):.3f}')
    print(f'Desviacion estandar de las salidas: {np.sqrt(np.var(yTrain)):.3f}')
    FILENAME = "bostonRegressor.sav"
    save(regressor, open(FILENAME, 'wb'))
    print(f"Guardado en {FILENAME}")
