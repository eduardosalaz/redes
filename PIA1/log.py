from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def logistic(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_theoretical = model.predict(x_test)
    err = 1 - accuracy_score(y_test, y_theoretical)
    return err

