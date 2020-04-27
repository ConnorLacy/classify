#!/usr/bin/python3

import os
import pickle
import logging
from utils.helper import importTestData
from utils.tester import Tester
from utils.preprocessor import Preprocessor

    
if __name__ == "__main__":

    pickleFiles = [
        # 'KNN_97_0426201952.pkl',
        'KNN_100_0427013428.pkl',
        'KNN_95_0426212533.pkl',
        'SVC_35_0426230110.pkl',
        'RF_97_0426211322.pkl',
        'RF_71_0426211600.pkl'
    ]

    models = []
    directory = os.path.join(os.getcwd(), 'models')
    for num, pickleFile in enumerate(pickleFiles, start=1):
        filePath = os.path.join(directory, f'Sample{num}', pickleFile)
        with open(filePath, 'rb') as file:
            print(f'Opening: {filePath}')
            models.append(pickle.load(file))

    tester = Tester()
    preprocessor = Preprocessor()
    for idx, model in enumerate(models, start=1):
        print(idx)
        logging.info('importing data')
        data = importTestData(idx)
        print(f'Data {idx}', data.head())
        # print(f'Data: {data_np}')
        # print(f'type: ', type(data_np))
        logging.info('cleaning data')
        data_mean_replacement = preprocessor.replaceMissingWithMean(data)
        # processed_data = preprocessor.select_features(data_mean_replacement)
        # newData = processed_data.to_numpy()
        # print(f'type: {type(processed_data)}')

        classifier = pickleFiles[idx-1].split('_')[0]
        logging.info('sending to tester')
        if classifier == 'KNN':
            tester.KNN(data_mean_replacement, model, idx)
        elif classifier == 'SVC':
            tester.SVC(data_mean_replacement, model, idx)
        elif classifier == 'RF':
            tester.random_forest(data_mean_replacement, model, idx)
        else:
            logging.error('Something went wrong')
    

