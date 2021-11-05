#!/usr/bin/env python3
r'''
 ./hw3.py
 Implements and tests a gradient descent algorithm with numerical
 differentiation.

 By        : Leomar Durán <https://github.com/lduran2/>
 When      : 2021-11-05t19:38
 Where     : Temple University
 For       : CIS 4526
 Version   : 1.1.0
 Dataset   : https://archive.ics.uci.edu/ml/datasets/wine+quality
 Canonical : https://github.com/lduran2/cis4526-machine_learning_foundations/blob/master/hw3-gradient_descent/hw3.py

 CHANGELOG :
    v1.1.0 - 2021-11-05t19:38
        split train, test data, calculated dimensionality
        

    v1.0.5 - 2021-11-05t18:56
        removing invalid rows

    v1.0.4 - 2021-11-05t17:49
        divided the features and labels using `np.split`

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

POSITIVE_RANGE = range(7,10)    # range for positive labels
NEGATIVE_RANGE = range(3,6)     # range for negative labels

def main():
    r'''
     classify the wine samples
     '''
    # read the dataset into a matrix
    dataset = np.genfromtxt(
        DATA_FILENAME, delimiter=DELIMITER,
        skip_header=True, dtype=np.float64)
    (num_rows, num_cols) = dataset.shape
    print(dataset.dtype.names)
    print(dataset)

    # divide into features and labels
    # function to classify the label scalars
    vec_classify = np.vectorize(classify)
    # split the dataset
    (unvalid_features, M_label_scalars) = \
        np.split(dataset, (num_cols - 1,), axis=1)
    # convert to a vector
    v_label_scalars = M_label_scalars.reshape((num_rows,))
    # classify the labels
    unvalid_labels = vec_classify(v_label_scalars)
    # remove all 0 rows
    # 0 rows are when the label scalar is neither in [7..10[ nor [3..6[
    should_keep_rows = (unvalid_labels != 0)
    print(should_keep_rows)
    features = unvalid_features[should_keep_rows,:]
    labels = unvalid_labels[should_keep_rows]
    print(features)
    print(labels)

    (num_examples,) = labels.shape          # num of valid examples
    num_test = (num_examples//4)            # num of test examples
    num_train = (num_examples - num_test)   # num of training examples
    num_dims = features.shape[1]            # dimensionality
    print(r'train:', num_train, r'; test:', num_test)

    # split features
    (train_x, test_x) = np.split(features, (num_train,), axis=0)
    # split labels
    (train_y, test_y) = np.split(labels, (num_train,), axis=0)
    print(train_x)
    print(test_x)
    print(train_y)
    print(test_y)
# end def main()

def classify(scalar):
    r'''
     Classifies a vector value as (+), (-) or ZERO
     @param scalar : 'int' = value to classify
     @return
        (+1) if the value is in [7..10[
        (-1) if the value is in [3..6[
        
     '''
    #
    result = 0  # default value to return
    if (scalar in POSITIVE_RANGE):
        result = 1
    # end if (scalar in POSITIVE_RANGE)
    elif (scalar in NEGATIVE_RANGE):
        result = -1
    # end elif (scalar in NEGATIVE_RANGE)
    return result
# end

# run main if this package was run
if (__name__==r'__main__'):
    main()
# end if (__name__==r'__main__')
