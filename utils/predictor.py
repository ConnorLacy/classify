#!/usr/bin/python3
import os
import sys
import pickle
import logging
import numpy as np
from datetime import datetime

from utils.helper import savePickle
from utils.helper import showGraph
from utils.helper import scaleData

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class Predictor():
    sampleNumber = None

    def __init__(self, test_size=0.2, random_state=27, verbose=False, save=False, acceptance=85):
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.acceptance = acceptance
        self.save = save

    def runAll(self, data, labels, neighbors, dataset, raw):
        self.sampleNumber = dataset
        if self.verbose:
            logging.info('Running test on all models\n')
        svc = self.SVC(data, labels)
        knn = self.KNN(data, labels, neighbors)
        bayes = self.naive_bayes(data, labels)
        if not raw:
            dtree = self.decision_tree(data, labels)
            randomForest = self.random_forest(data, labels)
            return svc, knn, bayes, dtree, randomForest

        return svc, knn, bayes
           

    def SVC(self, data, labels):
        # Partiton Training Data
        if self.verbose:
            logging.info('Implementing Support Vector Classifier...')
        train, test, train_labels, test_labels = train_test_split(
                data.values, labels, test_size=self.test_size, 
                random_state=self.random_state)

        train_scaled, test_scaled = scaleData(train, test)
        # print(f'Ravel shit SVC: {train_labels.values.ravel()}')
        SVC_model = SVC()
        SVC_model.fit(train_scaled, train_labels.values.ravel())
        SVC_prediction = SVC_model.predict(test_scaled)
        score = accuracy_score(SVC_prediction, test_labels)

        if self.verbose:
            logging.info(f'SVC Predictions Complete')
            logging.info('Score: {:0.2f}%\n'.format(score*100))
        if self.verbose == 2:
                logging.info(f'{confusion_matrix(SVC_prediction, test_labels)}')

        if score*100 >= self.acceptance and self.save:
            savePickle(SVC_model, 'SVC', score*100, self.sampleNumber)

        return score

    def KNN(self, data, labels, neighbors):
        # Partition Training Data 
        if self.verbose:
            logging.info(f'Implementing KNN Classifier with K={neighbors}...')
        train, test, train_labels, test_labels = train_test_split(
                data.values, labels, test_size=self.test_size, 
                random_state=self.random_state)

        # train_scaled, test_scaled = scaleData(train, test)
        train_scaled, test_scaled = train, test
        # _, testNoRavel = np.meshgrid(test, test)
        # print(f'Ravel shit KNN: {train_labels.values.ravel()}')

        # print(f'Train data {type(train)}')
        # print(f'New Ravel: {type(testNoRavel)}')
        KNN_model = KNeighborsClassifier(n_neighbors=neighbors)
        # print(f'train scaled: {train_scaled}\nravel label: {testNoRavel}')
        KNN_model.fit(train_scaled, train_labels.values.ravel())
        KNN_prediction = KNN_model.predict(test_scaled)
        score = accuracy_score(KNN_prediction,test_labels)

        if self.verbose:
            logging.info('KNN Predictions Complete')
            logging.info('Score: {:0.2f}%\n'.format(score*100))
        if self.verbose == 2:
            logging.info(f'KNN Classifier: \n\n{classification_report(KNN_prediction, test_labels)}')

        if score*100 >= self.acceptance and self.save:
            savePickle(KNN_model, 'KNN', score*100, self.sampleNumber)

        return score

    def naive_bayes(self, data, labels):
        if self.verbose:
            logging.info(f'Implementing Naive Bayes Classifier...')
        train, test, train_labels, test_labels = train_test_split(
                data.values, labels, test_size=self.test_size, 
                random_state=self.random_state)
        
        train_scaled, test_scaled = scaleData(train, test)
        
        gnb = GaussianNB()
        model = gnb.fit(train_scaled, train_labels.values.ravel())
        prediction = gnb.predict(test_scaled)
        score = accuracy_score(test_labels, prediction)

        if self.verbose:
            logging.info(f'GNB Predictions Complete')
            logging.info('Score: {:0.2f}%\n'.format(score*100))
        if score*100 >= self.acceptance:
            savePickle(gnb, 'GNB', score*100, self.sampleNumber)

        return score

    def decision_tree(self, data, labels):
        if self.verbose:
            logging.info(f'Implementing Decision Tree Classifier...')
        train, test, train_labels, test_labels = train_test_split(
                data.values, labels, test_size=self.test_size, 
                random_state=self.random_state)

        train_scaled, test_scaled = scaleData(train, test)

        dt = DecisionTreeClassifier(random_state=0, max_depth=2)
        model = dt.fit(train_scaled, train_labels.values.ravel())
        prediction  = dt.predict(test_scaled)
        score = accuracy_score(test_labels, prediction)

        if self.verbose:
            logging.info(f'DT Predictions Complete')
            logging.info('Score: {:0.2f}%\n'.format(score*100))
        if score*100 >= self.acceptance and self.save:
            savePickle(dt, 'DT', score*100, self.sampleNumber)

        return score

    def random_forest(self, data, labels):
        if self.verbose:
            logging.info('Implementing Random Forest Classifier...')
        train, test, train_labels, test_labels = train_test_split(
                        data.values, labels, test_size=self.test_size, 
                        random_state=self.random_state)

        train_scaled, test_scaled = scaleData(train, test)
        # print(f'Ravel shit RF: {train_labels.values.ravel()}\n')
        # print(f'Train type: {type(train)}')
        # print(f'Ravel type: {type(train_labels.values.ravel())}')

        rfc = RandomForestClassifier(random_state=101)
        # rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
        rfecv = RFECV(estimator=rfc, step=1, cv=7, scoring='accuracy')
        rfecv.fit(train_scaled, train_labels.values.ravel())

        if self.verbose:
            logging.info('Optimal number of features: {}'.format(rfecv.n_features_))
        if self.verbose == 2: 
            showGraph(rfecv.grid_scores_)
            
        prediction = rfecv.predict(test_scaled)
        score = accuracy_score(test_labels, prediction)

        if self.verbose:
            logging.info(f'RF Predictions Complete')
            logging.info('Score: {:0.2f}%\n'.format(score*100))
        if score*100 >= self.acceptance and self.save:
            savePickle(rfecv, 'RF', score*100, self.sampleNumber)

        return score
        import numpy as np
