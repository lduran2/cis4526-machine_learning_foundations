#!/usr/bin/env python3
r'''
 ./hw3.py
 Implements and tests a gradient descent algorithm with numerical
 differentiation.

 By        : Leomar Durán <https://github.com/lduran2/>
 When      : 2021-11-05t21:27
 Where     : Temple University
 For       : CIS 4526
 Version   : 1.2.0
 Dataset   : https://archive.ics.uci.edu/ml/datasets/wine+quality
 Canonical : https://github.com/lduran2/cis4526-machine_learning_foundations/blob/master/hw3-gradient_descent/hw3.py

 CHANGELOG :
    v1.2.0 - 2021-11-05t21:27
        folded in the template

    v1.1.1 - 2021-11-05t21:03
        abstracted the parts of the main method

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

DATA_FILENAME = r'winequality-white.csv'  # name of file holding dataset
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
    # get the training and testing data
    (features, labels) = splitFeaturesLabels(dataset)
    (train_x, test_x, train_y, test_y, \
            num_test, num_train, num_dims) = \
        splitTrainTest(features, labels)
    #
# end def main()

def hinge_loss(train_y, pred_y):
    r'''
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param pred_y : 'numpy.ndarray' = predicted labels (num_train×1)
     '''
    L = None
    return L
# end def hinge_loss(train_y, pred_y)

def squared_loss(train_y, pred_y):
    r'''
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param pred_y : 'numpy.ndarray' = predicted labels (num_train×1)
     '''
    L = None
    return L
# end def squared_loss(train_y, pred_y)

def logistic_loss(train_y, pred_y):
    r'''
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param pred_y : 'numpy.ndarray' = predicted labels (num_train×1)
     '''
    L = None
    return L
# end def logistic_loss(train_y, pred_y)

def l1_reg(w):
    r'''
     @param w : 'numpy.ndarray' =
        vector of linear classifier weights
        (num_dims×1)
     '''
    L = None
    return L
# end def l1_reg(w)

def l2_reg(w):
    r'''
     @param w : 'numpy.ndarray' =
        vector of linear classifier weights
        (num_dims×1)
     '''
    L = None
    return L
# end def l2_reg(w)

def train_classifier(train_x, train_y, learn_rate, loss, \
        lambda_val, regularizer):
    r'''
     @param train_x : 'numpy.ndarray' = training features (num_train×1)
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param learn_rate : 'float' = the learning rate for gradient descent
     @param loss : 'function' = a loss function
     @param lambda_val : 'float' = tradeoff parameter
     @param regularizer : 'function' = a regularizer function
     @return the vector of learned linear classifier weights
     '''
    return None
# end def train_classifier(train_x, train_y, learn_rate, loss,
#       lambda_val, regularizer)

def test_classifier(w, test_x):
    r'''
     @param w : 'numpy.ndarray' =
        vector of linear classifier weights
        (num_dims×1)
     @param test_x : 'numpy.ndarray' = data matrix (num_test × num_dims)
     @return the num_test×1 prediction vector
     '''
    pred_y = None
    return pred_y
# end def test_classifier(w, test_x)

def compute_accuracy(test_y, pred_y):
    r'''
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param pred_y : 'numpy.ndarray' = predicted labels (num_train×1)
     @return the classification accuracy in [0.0, 1.0]
     '''
    acc = None
    return acc
# end def compute_accuracy(test_y, pred_y)

def get_id():
    r'''
     @return my TempleU email
     '''
    return r'tuh24865'
# end def get_id()

def splitFeaturesLabels(dataset):
    r'''
     Divides the dataset into features and labels.
     @param dataset : 'numpy.ndarray' = the dataset to divide
     @return a tuple containing the features and the labels
     '''
    # get the number of rows and columns
    (num_rows, num_cols) = dataset.shape

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
    features = unvalid_features[should_keep_rows,:]
    labels = unvalid_labels[should_keep_rows]

    return (features, labels)
# end def splitFeaturesLabels(dataset)

def splitTrainTest(features, labels):
    r'''
     Divides the train, test data × features, labels.
     @param features : 'numpy.ndarray' = of the dataset
     @param labels : 'numpy.ndarray' = of the dataset
     @return a tuple containing the (train_x, test_x, train_y, test_y),
        the number of testing and training examples, and the
        dimensionality
     '''
    (num_examples,) = labels.shape          # num of valid examples
    num_test = (num_examples//4)            # num of test examples
    num_train = (num_examples - num_test)   # num of training examples
    num_dims = features.shape[1]            # dimensionality

    # split features
    (train_x, test_x) = np.split(features, (num_train,), axis=0)
    # split labels
    (train_y, test_y) = np.split(labels, (num_train,), axis=0)

    return (train_x, test_x, train_y, test_y, \
        num_test, num_train, num_dims)
# end def splitTrainTest(features, labels)

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
# end def classify(scalar)

# run main if this package was run
if (__name__==r'__main__'):
    main()
# end if (__name__==r'__main__')
