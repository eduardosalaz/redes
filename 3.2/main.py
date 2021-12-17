import time

import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential

import mnistKeras
import mnistLogistic
import mnistMLP

if __name__ == '__main__':
    # leyendo la base de datos MNIST
    data = np.loadtxt('mnist_train.csv', delimiter=',')
    print("Lectura de la base de datos completa")
    ncol = data.shape[1]
    # definiendo entradas y salidas
    X = data[:, 1:ncol]
    y = data[:, 0]

    data = np.loadtxt('mnist_test.csv', delimiter=',')
    # print(data)
    ncol = data.shape[1]
    # definiendo entradas y salidas
    X_test = data[:, 1:ncol]
    y_test = data[:, 0]

    (x_train, y_train), (x_test, yTest) = datasets.mnist.load_data()

    # normalize
    xTrain = x_train.astype("float32") / 255
    xTest = x_test.astype("float32") / 255

    # classes
    yTrain = utils.to_categorical(y_train, 10)
    yTest = utils.to_categorical(yTest, 10)

    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.SpatialDropout2D(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss=losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

    totalTimeLog = 0
    totalTimeMLP = 0
    totalTimeKeras = 0
    totalErrLog = 0
    totalErrMLP = 0
    totalErrKeras = 0

    for i in range(5):
        tic = time.perf_counter()
        errLogistic = mnistLogistic.logistic(X, y, X_test, y_test)
        toc = time.perf_counter()
        timeLog = toc - tic
        print(f"Tiempo de procesador para el entrenamiento Logistico (seg):{timeLog}")
        totalTimeLog += timeLog
        totalErrLog += errLogistic

        tic = time.perf_counter()
        errMLP = mnistMLP.mlp(X, y, X_test, y_test)
        toc = time.perf_counter()
        timeMLP = toc - tic
        print(f"Tiempo de procesador para el entrenamiento MLP (seg):{timeMLP}")
        totalTimeMLP += timeMLP
        totalErrMLP += errMLP

        timeKeras, errKeras = mnistKeras.keras(model, xTrain, yTrain, xTest, yTest)
        totalTimeKeras += timeKeras
        totalErrKeras += errKeras

    print(f"Tiempo Promedio de Logistic: {totalTimeLog / 5:.3f}")
    print(f"Tiempo Promedio de MLP: {totalTimeMLP / 5:.3f}")
    print(f"Tiempo Promedio de Keras: {totalTimeKeras / 5:.3f}")

    print(f"Error Promedio de Logistic: {totalErrLog / 5:.3f}")
    print(f"Error Promedio de MLP: {totalErrMLP / 5:.3f}")
    print(f"Error Promedio de Keras: {totalErrKeras / 5:.3f}")

