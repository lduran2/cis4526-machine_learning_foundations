#!/usr/bin/env python3
r'''
 ./hw3.py
 Implements and tests a gradient descent algorithm with numerical
 differentiation.

 By        : Leomar Durán <https://github.com/lduran2/>
 When      : 2021-11-05t17:49
 Where     : Temple University
 For       : CIS 4526
 Version   : 1.0.4
 Dataset   : https://archive.ics.uci.edu/ml/datasets/wine+quality
 Canonical : https://github.com/lduran2/cis-4526-hw2-kNN-vs-pocket/blob/master/hw2.py

 CHANGELOG :
    v1.0.4 - 2021-11-05t17:49
        divided the features and labels using np.split

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
    (N_ROWS, N_COLS) = dataset.shape
    print(dataset.dtype.names)
    print(dataset)

    # divide into features and labels
    # function to classify the label scalars
    vec_classify = np.vectorize(classify)
    # split the dataset
    (features, M_label_scalars) = np.split(dataset, (N_COLS - 1,), axis=1)
    # convert to a vector
    v_label_scalars = M_label_scalars.reshape((N_ROWS,))
    # classify the labels
    labels = vec_classify(v_label_scalars)
    print(labels)
# end def main()

def classify(scalar):
    r'''
     Classifies a vector value as (+) or (-)
     @param scalar : 'int' = value to classify
     @return +1 if in range(7,10) else -1
     '''
    #
    result = -1
    if (scalar in range(7,10)):
        result = 1
    # end if (scalar in range(7,10))
    return result
# end

# run main if this package was run
if (__name__==r'__main__'):
    main()
# end if (__name__==r'__main__')
