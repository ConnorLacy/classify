#!/usr/bin/python3

import sys
from time import sleep
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

if __name__ == "__main__":
    print('Attempting to load features into pandas df...')
    sleep(1)
    data = pd.read_csv("../data/classification/iris.csv", delimiter=",")
    print(data.head(5))
    X = data.iloc[:,:-1].values
    y = data['species']
    print(f'\ndata values\n{X}')
    print(f'\nlabels\n{y}')

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=27)
    print(f'X_train: \n{X_train}')
    print(f'y_train: \n{y_train}')

    SVC_model = SVC()
    KNN_model = KNeighborsClassifier(n_neighbors=5)

    SVC_model.fit(X_train, y_train)
    KNN_model.fit(X_train, y_train)

    SVC_prediction = SVC_model.predict(X_test)
    KNN_prediction = KNN_model.predict(X_test)

    # Accuracy score is the simplest way to evaluate
    print(accuracy_score(SVC_prediction, y_test))
    print(accuracy_score(KNN_prediction, y_test))

    # But Confusion Matrix and Classification Report give more details about performance
    print(confusion_matrix(SVC_prediction, y_test))
    print(classification_report(KNN_prediction, y_test))