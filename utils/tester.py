#!/usr/bin/python3
import os
import sys
import pickle
import logging
from datetime import datetime

from utils.helper import savePickle
from utils.helper import showGraph
from utils.helper import scaleData
from utils.helper import writeTest

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

class Tester():

    def SVC(self, data, model, setNumber):
        # Partiton Training Data
        logging.info('Implementing Support Vector Classifier...')
        # prediction = model.predict(data)
        # logging.info(f'SVC Test Complete')
        # writeTest(prediction, setNumber)
        # write out

    def KNN(self, data, model, setNumber):
        # Partition Training Data 
        logging.info(f'Implementing KNN Classifier...')
        prediction = model.predict(data)
        logging.info('KNN Predictions Complete')
        writeTest(prediction, setNumber)

    def random_forest(self, data, model, setNumber):
        logging.info('Implementing Random Forest Classifier...')
        # prediction = model.predict(data)
        # logging.info(f'RF Predictions Complete')
        # writeTest(prediction, setNumber)
        