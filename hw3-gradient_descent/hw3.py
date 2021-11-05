#!/usr/bin/env python3
r'''
 ./hw3.py
 Implements and tests a gradient descent algorithm with numerical
 differentiation.

 By        : Leomar Durán <https://github.com/lduran2/>
 When      : 2021-11-05t17:27
 Where     : Temple University
 For       : CIS 4526
 Version   : 1.0.3
 Dataset   : https://archive.ics.uci.edu/ml/datasets/wine+quality
 Canonical : https://github.com/lduran2/cis-4526-hw2-kNN-vs-pocket/blob/master/hw2.py

 CHANGELOG :
    v1.0.3 - 2021-11-05t17:27
        extracted the labels (using matrix multiplication)

    v1.0.2 - 2021-11-05t16:31
        now using numpy

    v1.0.1 - 2021-11-05t15:44
        named the features

    v1.0.0 - 2021-11-05t15:29
        printing the rows of the data file
 '''
#

import numpy as np

DATA_FILENAME = r'winequality-white.csv'  # name of file holding the dataset
DELIMITER = ';' # used to separate values in DATA_FILENAME

def main():
    r'''
     classify the wine samples
     '''
    # read the dataset into a matrix
    dataset = np.genfromtxt(DATA_FILENAME, delimiter=DELIMITER, skip_header=True, dtype=np.float64)
    N_COLS = dataset.shape[1]
    print(dataset.dtype.names)
    print(dataset)

    # get the labels
    # function to classify the label scalars
    classify = np.vectorize(lambda x : (+1) if (x in range(7,10)) else (-1))
    # ZERO rows to remove features
    label_mcand_features = np.zeros(((N_COLS - 1),))
    # ONE row to keep the label
    label_mcand_one = np.array([1])
    # concatenate ZERO and ONE rows to a vector
    label_mcand = np.concatenate(
        (label_mcand_features, label_mcand_one), axis=0)
    # multiply the dataset and labels
    labels_scalars = (dataset @ label_mcand)
    # and classify
    labels = classify(labels_scalars)
    print(labels)
# end def main()

# run main if this package was run
if (__name__==r'__main__'):
    main()
# end if (__name__==r'__main__')
