#!/usr/bin/python3
import sys
import logging
import argparse
import warnings

from utils.helper import importSamples
from utils.helper import calculateDelta
from utils.preprocessor import Preprocessor
from utils.predictor import Predictor

import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore', 'Increase *')
warnings.filterwarnings('ignore', '667: UserWarning:*')

parser = argparse.ArgumentParser(description='Perform ML Test on Datasets')
parser.add_argument(
        '--sample',
        type=int,
        nargs='+',
        required=False,
        help='Specifies the Sample(s) you wish to run the test on\n')
parser.add_argument(
        '--knn',
        type=int,
        default=3,
        help='Enter the number of neighbors for KNN\n')
parser.add_argument(
        '-v',
        '--verbose',
        type=int,
        default=0,
        choices=[0,1,2],
        help='Display verbose console messages on what is happening\n')
parser.add_argument(
        '--all',
        action='store_true',
        default=False,
        help='Run each of the classifier tests on each dataset\n')
parser.add_argument(
        '--acceptance',
        type=int,
        default=85,
        help='Specify the accuracy percentage deemed acceptable for test\n')
parser.add_argument(
        '-p',
        '--preprocess',
        type=int,
        default=0,
        help='Toggle data preprocessing technique\n')
parser.add_argument(
        '--save',
        action='store_true',
        help='This will save the model if the accuracy is above the threshold.\n')
args = parser.parse_args()

def runTests(dataset):
    data, labels = importSamples(dataset, args.verbose)
    print(f'run test data: ', data.head())
    print(f'type: ', type(data))

    # Run analysis before preprocessing
    if args.verbose:
        logging.info('Performing analysis on raw data...')
    raw_SVC, raw_KNN, raw_GNB = predictor.runAll(
            data, labels, args.knn, dataset, raw=True)
    
    # Run Analysis after preprocessing
    if args.verbose:
        logging.info('Performing analysis on preprocessed data...\n')
    # Replace NaNs with mean value ( Impute )
    data_mean_replacement = preprocessor.replaceMissingWithMean(data)
    if args.preprocess == 1:
        processed_data = preprocessor.select_features(data_mean_replacement)
        processed_SVC, processed_KNN, processed_GNB, processed_DT, processed_RF = predictor.runAll(
            processed_data, labels, args.knn, dataset, raw=False)
    else:
        processed_SVC, processed_KNN, processed_GNB, processed_DT = predictor.runAll(
            data_mean_replacement, labels, args.knn, dataset, raw=False)
        
    # Determine improvment by preprocessing technique
    improvement_SVC = calculateDelta(raw_SVC, processed_SVC)
    improvement_KNN = calculateDelta(raw_KNN, processed_KNN)
    improvement_GNB = calculateDelta(raw_GNB, processed_GNB)

    if args.verbose:
        summarySVC = createSummary(
            'Support Vector Classifier',raw_SVC, 
            processed_SVC, improvement_SVC)
        summaryKNN = createSummary(
            'KNN Classifier', raw_KNN,
            processed_KNN, improvement_KNN)
        summaryGNB = createSummary(
            'GNB Classifier', raw_GNB,
            processed_GNB, improvement_GNB)
        summaryDT = createSummary(
            'DT Classifier', 0, processed_DT, 0, False)
        summaryRF = createSummary(
            'RF Classifier', 0, processed_RF, 0, False)

        print('\n' + '='*30, f'Summary of Sample {dataset}', '='*30)
        print(summarySVC)
        print(summaryKNN)
        print(summaryGNB)
        print(summaryDT)
        print(summaryRF)
        print('\n' + '='*80, '\n')

    rawArr = [
        round(raw_SVC*100, 2),
        round(raw_KNN*100, 2),
        round(raw_GNB*100, 2),
        0,
        0
    ]
    processedArr = [
        round(processed_SVC*100, 2),
        round(processed_KNN*100, 2),
        round(processed_GNB*100, 2),
        round(processed_DT*100, 2),
        round(processed_RF*100, 2)
    ]
    
    return rawArr, processedArr

def createSummary(classifier, rawScore, processedScore, improvement, hadRawTest=True):
    acceptable = 'Yes' if processedScore*100>(args.acceptance) else 'No'
    if not hadRawTest:
        summary = (f'\n{classifier} Accuracy:\n' +
        '  > Before preprocessing: {:0.2f}%'.format(rawScore*100) + 
        '\n  > After preprocessing: {:0.2f}%'.format(processedScore*100) +
        '\n  > Improvement: {:0.2f}%'.format(improvement) +
        '\n  > Acceptable? {}'.format(acceptable))
    else:
        summary = (f'\n{classifier} Accuracy:\n' +
        '  > Before preprocessing: N/A' + 
        '\n  > After preprocessing: {:0.2f}%'.format(processedScore*100) +
        '\n  > Acceptable? {}'.format(acceptable))
    return summary


if __name__ == "__main__":
    predictor = Predictor(
        test_size=0.2, random_state=27, 
        verbose=args.verbose, save=args.save, 
        acceptance=args.acceptance)
    preprocessor = Preprocessor(verbose=args.verbose)
    samples = [1,2,3,4,5] if args.all else args.sample

    raw = []
    processed = []
    for i in samples:
        raw_res, processed_res = runTests(i)
        raw.append(raw_res)
        processed.append(processed_res)

    if not args.verbose:
        print('Format: [SVC, KNN, GNB, DTREE]\n')
        for i in range(len(raw)):
            print(f'Sample {samples[i]}:', 
            f'\n  > Raw: {raw[i]}', 
            f'\n  > Processed: {processed[i]}\n')
