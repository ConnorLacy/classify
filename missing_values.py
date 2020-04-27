#!/usr/bin/python3
import argparse
import logging

from utils.predictor import Predictor
from utils.preprocessor import Preprocessor
from utils.helper import importMissing
from utils.helper import saveMissing

parser = argparse.ArgumentParser(description='Perform ML Test on Datasets')
parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='Display verbose console messages on what is happening\n')
parser.add_argument(
        '-sv',
        '--save',
        action='store_true',
        help='This will save the model if the accuracy is above the threshold.\n')
parser.add_argument(
        '--sample',
        type=int,
        nargs='+',
        required=False,
        help='Specifies the Sample(s) you wish to run the test on\n')
parser.add_argument(
        '-k',
        '--knn',
        type=int,
        required=False,
        default=3,
        help='Specifies the Sample(s) you wish to run the test on\n')
parser.add_argument(
        '-w',
        '--weight',
        type=str,
        required=False,
        default='uniform',
        choices=['uniform', 'distance'],
        help='Specifies the Sample(s) you wish to run the test on\n')
args = parser.parse_args()

def estimateMissing(dataset_number):
    data = importMissing(dataset_number, args.verbose)

    # Run analysis before preprocessing
    logging.info('Performing analysis on raw data...')
    
    dataNaN, count = preprocessor.convertMissingValueToNan(data)
    logging.info(f'Replaced {count} values with NaN')
    imputed = preprocessor.impute_knn(dataNaN, args.knn, args.weight)
    logging.info(f'Imputed {count} values with KNN imputation')
    if(args.save):
        saveMissing(dataset_number, imputed)


if __name__ == "__main__":
    predictor = Predictor(
        test_size=0.2, random_state=27, 
        verbose=args.verbose, save=args.save)
    preprocessor = Preprocessor(verbose=args.verbose)
    samples = [1,2] if not args.sample else args.sample
    for i in samples:
        estimateMissing(i)