# Este programa crea una red neuronal profunda de tensorflow usando Keras

import time
import pickle

from tensorflow.keras import callbacks


def keras(model, xTrain, yTrain, xTest, yTest):
    # callbacks
    earlyStop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
    reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=4, factor=0.5, min_lr=1e-6,
                                           verbose=1)
    tic = time.perf_counter()
    print("Entrenando Red Neuronal: ")
    model.fit(xTrain, yTrain, batch_size=128, epochs=12, callbacks=[earlyStop, reduceLR], shuffle=True,
              validation_split=0.2, verbose=0) # no output
    toc = time.perf_counter()
    score = model.evaluate(xTest, yTest, verbose=0)
    print(f"Validation loss:{score[0]}")
    print(f"Validation accuracy: {score[1]}")
    totalTime = toc - tic
    print(f"Tiempo de procesador para el entrenamiento Keras (seg):{totalTime}")
    print(toc - tic)
    filename = 'finalized_modelKeras.sav'
    pickle.dump(model, open(filename, 'wb'))
    return totalTime, score[0]



