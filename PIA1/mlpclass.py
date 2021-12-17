from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def mlp_class(x_train, y_train, x_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100, 120, 100), max_iter=2000)
    model.fit(x_train, y_train)
    y_theoretical = model.predict(x_test)
    err = 1 - accuracy_score(y_test, y_theoretical)
    return err
