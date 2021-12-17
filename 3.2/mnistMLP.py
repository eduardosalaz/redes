# Este programa crea una MLP classifier de sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc
import pickle


def mlp(X, y, X_test, y_test):
    clf = MLPClassifier(solver='lbfgs', max_iter=500, activation='logistic', hidden_layer_sizes=(20, 20),
                        verbose=False, random_state=10)  # seed for reproducibility
    clf.fit(X, y)
    print("Entrenamiento completo de MLP: ")
    predicted = clf.predict(X_test)
    err = 1 - acc(y_test, predicted)
    print(f'Error del {err:.3f}')
    filename = 'finalized_modelMLP.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return err
