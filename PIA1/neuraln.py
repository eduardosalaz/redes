import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential


def deep_nn(x_train, y_train, x_test, y_test):
    y_train = utils.to_categorical(y_train, num_classes=2)
    y_test = utils.to_categorical(y_test, num_classes=2)
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(8,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128, verbose=0)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "validation loss"], loc="upper left")
    plt.show()
    val_acc = history.history['val_accuracy'][4]
    err = 1 - val_acc
    return err



