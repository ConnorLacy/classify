import os
import pickle
import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, PolynomialFeatures, PowerTransformer, StandardScaler
import numpy as np

def importSamples(dataset_number, verbose):
    delimeter_dict = {
        1 : "\t",
        2 : "\s+",
        3 : "\t",
        4 : "\s+",
        5 : "\s+",
        6 : "\s+"
    }
    
    filePath = os.path.join(os.getcwd(), 'data', 'classification', 'train')
    delimeter = delimeter_dict[dataset_number]
    logging.info(f"Running tests on Sample {dataset_number}\n")
    if verbose:
        logging.info(f"Using Data from 'TrainData{dataset_number}.txt'") 
        logging.info(f"Using Labels from 'TrainLabel{dataset_number}.txt'")
    data = pd.read_csv(filePath + f'/TrainData{dataset_number}.txt',
                    header=None, delimiter=delimeter)
    labels = pd.read_csv(filePath + f'/TrainLabel{dataset_number}.txt', header=None)
    return data, labels

def importMissing(dataset_number, verbose):
    delimeter = "\t"
    filePath = os.path.join(os.getcwd(), 'data', 'missing_value_est')
    logging.info(f"Estimating values on Sample {dataset_number}\n")
    if verbose:
        logging.info(f"Using Data from 'MissingData{dataset_number}.txt'") 
    data = pd.read_csv(filePath + f'/MissingData{dataset_number}.txt',
                    header=None, delimiter=delimeter)
    return data

def savePickle(model, modelType, score, sampleNumber):
    score = int(round(score))
    directory = os.path.join(os.getcwd(), 'models', f'Sample{sampleNumber}')
    filename = f'{modelType}_{score}_{datetime.now().strftime("%m%d%H%M%S")}.pkl'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), 'wb') as file:
        pickle.dump(model, file)

def saveMissing(dataset_number, data):
    filename = f'LacyMissingResult{dataset_number}.txt'
    dir = os.path.join(os.getcwd(), 'data', 'missing_value_est', 'imputed')
    if not os.path.exists(dir):
        os.mkdir(dir)
    fullPath = os.path.join(dir, filename)
    np.savetxt(fullPath, data, fmt='%1.15f', delimiter="\t")  
    logging.info('Saved file')
        
    
def calculateDelta(rawScore, preprocessedScore):
    delta = ((preprocessedScore-rawScore)/rawScore)*100
    return delta

def showGraph(classifierGridScores):
    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(classifierGridScores) + 1), classifierGridScores, color='#303F9F', linewidth=3)
    plt.show()

def scaleData(train, test):
    scaler = MinMaxScaler()
    # scaler = PolynomialFeatures()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled

def writeTest(prediction, setNumber):
    fileName = f'LacyTestData{setNumber}.txt'
    directory = os.path.join(os.getcwd(), 'data', 'classification', 'results')
    if not os.path.exists(directory):
        os.mkdir(directory)
    filePath = os.path.join(directory, fileName)
    # prediction.to_csv(filePath, header=None, index=None, sep='\t', mode='w')
    np.savetxt(filePath, prediction, fmt='%d', delimiter="\t")  
    logging.info('Wrote test data')

def importTestData(dataset_number):
    delimeter_dict = {
        1 : "\t",
        2 : "\s+",
        3 : ",",
        4 : "\s+",
        5 : "\s+",
    }
    filePath = os.path.join(os.getcwd(), 'data', 'classification', 'test')
    delimeter = delimeter_dict[dataset_number]
    logging.info(f"Using Data from 'TestData{dataset_number}.txt'") 
    data = pd.read_csv(filePath + f'/TestData{dataset_number}.txt',
                    header=None, delimiter=delimeter)
    return data