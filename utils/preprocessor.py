#!/usr/bin/python3
import sys
import numpy
import logging
from sklearn.impute import KNNImputer
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class Preprocessor():
    def __init__(self, verbose=False):
        self.verbose = verbose

    def replaceMissingWithMean(self, data):
        data_with_NaNs, count = self.convertMissingValueToNan(data)
        data_with_NaNs.fillna(data_with_NaNs.mean(), inplace=True)
        if self.verbose == 2:
            logging.info(f'Imputed {count} NaN values by Mean Value')
        return data_with_NaNs
    
    def convertMissingValueToNan(self, data):
        data_copy = data.copy()
        npArr = data.to_numpy() 
        count = 0
        for i, row in enumerate(npArr):
            for j, val in enumerate(row):
                if type(val) == numpy.float64 and val.item() >= 1.00000000000000e+98:
                    data_copy.at[i, j] = numpy.nan
                    count += 1
        if self.verbose == 2:
            logging.info(f'Data with NaN:\n{data_copy}')
        if self.verbose == 2:
            logging.info(f'Replaced {count} values with NaN')
        return data_copy, count
    
    def select_features(self, data):
        if self.verbose:
            logging.info(f'Implementing Feature Selection')
        correlated_features = set()
        correlation_matrix = data.corr()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        if self.verbose:
            if correlated_features != set():
                logging.info(f'Correlated features: {correlated_features}')
            else:
                logging.info('No correlated features detected')
        processed_data = data.drop(correlated_features, axis=1)
        return processed_data

    def impute_knn(self, data, neighbors, weights):
        data_copy = data.copy()
        imputer = KNNImputer(n_neighbors=neighbors, weights=weights, missing_values=np.NaN)
        result = imputer.fit_transform(data_copy)
        return result