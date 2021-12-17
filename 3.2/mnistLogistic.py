# Este programa entrena una regresion logistica para MNIST y guarda el modelo entrenado para posterior uso.
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle


def logistic(X, y, X_test, y_test):
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    print("Entrenamiento completo de Regresión Logística:")
    # predecir los valores de X_test
    predicted = clf.predict(X_test)
    # para finalizar se calcula el error
    err = 1 - accuracy_score(y_test, predicted)
    print(f'Error de: {err:.3f}')
    # el modelo entrenado se salva en disco
    filename = 'finalized_modelLog.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return err
