from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pandas import read_csv as csv
import numpy as np
from pickle import dump as save

if __name__ == '__main__':
    xTrain = csv('xsonarTrain.csv')
    yTrain = np.ravel(csv('ysonarTrain.csv'))
    xTest = csv('xsonarTest.csv')
    yTest = csv('ysonarTest.csv')

    classifier = MLPClassifier(solver='lbfgs', max_iter=5000, hidden_layer_sizes=(120, 120, 120))
    classifier.fit(xTrain.values, yTrain)
    yTheoretical = classifier.predict(xTest.values)  # we want the values per se, there are no feature names
    err = 1 - accuracy_score(yTest, yTheoretical)
    print(f"% de error del: {err * 100:.3f}")
    FILENAME = "sonarClassifier.sav"
    save(classifier, open(FILENAME, 'wb'))
    print(f"Guardado en {FILENAME}")
