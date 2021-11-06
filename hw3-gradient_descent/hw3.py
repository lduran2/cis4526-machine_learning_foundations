#!/usr/bin/env python3
r'''
 ./hw3.py
 Implements and tests a gradient descent algorithm with numerical
 differentiation.

 By        : Leomar Durán <https://github.com/lduran2/>
 When      : 2021-11-05t23:01
 Where     : Temple University
 For       : CIS 4526
 Version   : 1.4.1
 Dataset   : https://archive.ics.uci.edu/ml/datasets/wine+quality
 Canonical : https://github.com/lduran2/cis4526-machine_learning_foundations/blob/master/hw3-gradient_descent/hw3.py

 CHANGELOG :
    v1.4.1 - 2021-11-05t23:01
        tested the perceptron and calculated its accuracy

    v1.4.0 - 2021-11-05t22:40
        implemented perception

    v1.3.0 - 2021-11-05t22:21
        normalized features

    v1.2.1 - 2021-11-05t21:46
        abstracted train ratio and classifier

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

TRAIN_RATIO = 3                 # ratio of training to testing data

def main():
    r'''
     classify the wine samples
     '''
    # read the dataset into a matrix
    dataset = np.genfromtxt(
        DATA_FILENAME, delimiter=DELIMITER,
        skip_header=True, dtype=np.float64)
    # get the training and testing data
    (unnormal_features, labels) = splitFeaturesLabels(dataset, classify)
    features = normalizeFeatures(unnormal_features)
    (train_x, test_x, train_y, test_y, \
            num_test, num_train, num_dims) = \
        splitTrainTest(features, labels, TRAIN_RATIO)
    #

    # train, test the classifier with the data, and calculate accuracy
    w = train_classifier(train_x, train_y, 1, hinge_loss, 1, l1_reg)
    pred_y = test_classifier(w, test_x)
    acc = compute_accuracy(test_y, pred_y)
    print(acc)
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
     Implements a perceptron using the given training data.
     @param train_x : 'numpy.ndarray' = training features (num_train×1)
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param learn_rate : 'float' = the learning rate for gradient descent
     @param loss : 'function' = a loss function
     @param lambda_val : 'float' = tradeoff parameter
     @param regularizer : 'function' = a regularizer function
     @return the vector of learned linear classifier weights
     '''
    max_iter = 10               # maximum num of iterations
    num_dims = train_x.shape[1] # dimensionality
    w = np.zeros((num_dims,))   # zero out a vector (b = w[0])
    for k in range(max_iter):
        for (x,y) in zip(train_x, train_y):
            activation = w * x
            for d in range(num_dims):
                if ((y*activation[d]) <= 0):
                    w[d] += y*x[d]
                # end if ((y*activation) <= 0)
            # end for d in range(num_dims)
        # end for (x,y) in zip(train_x, train_y)
    # for k in range(0,max_iter)
    return w
# end def train_classifier(train_x, train_y, learn_rate, loss,
#       lambda_val, regularizer)

def test_classifier(w, test_x):
    r'''
     Tests the training model represented by `w` against test input
     `test_x`.
     @param w : 'numpy.ndarray' =
        vector of linear classifier weights
        (num_dims×1)
     @param test_x : 'numpy.ndarray' = data matrix (num_test × num_dims)
     @return the num_test×1 prediction vector
     '''
    pred_y = np.sign(test_x @ w)    # y = sgn(X.w)
    return pred_y
# end def test_classifier(w, test_x)

def compute_accuracy(test_y, pred_y):
    r'''
     Calculates the accuracy of a training model resulting in
     `pred_y` against test output `test_y`.
     @param train_y : 'numpy.ndarray' = training labels (num_train×1)
     @param pred_y : 'numpy.ndarray' = predicted labels (num_train×1)
     @return the classification accuracy in [0.0, 1.0]
     '''
    num_test = test_y.shape[0]          # num of test examples
    equals = np.equal(test_y, pred_y)   # true if test_y == pred_y
    acc = np.sum(equals)/num_test       # Acc = sum(1[equal])/nu(test_y)
    return acc
# end def compute_accuracy(test_y, pred_y)

def get_id():
    r'''
     @return my TempleU email
     '''
    return r'tuh24865'
# end def get_id()

def splitFeaturesLabels(dataset, classify):
    r'''
     Divides the dataset into features and labels.
     @param dataset : 'numpy.ndarray' = the dataset to divide
     @param classify : 'function' = label classifier function
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

def normalizeFeatures(unnormal_features):
    r'''
     Normalizes the given features
     @param unnormal_features : 'numpy.ndarray' = to normalize
     @return the given features normalized
     '''
    # center the data
    means = unnormal_features.mean(axis=0)
    centered = (unnormal_features - means)
    # scale the data
    absmaxa = np.amax(np.absolute(centered), axis=0)
    scaled = (centered / absmaxa)
    # 1 pad
    num_examples = scaled.shape[0]
    one_pad = np.ones((num_examples,1))
    padded = np.concatenate((one_pad, scaled), axis=1)
    return padded
# end def normalizeFeatures()

def splitTrainTest(features, labels, train_ratio):
    r'''
     Divides the train, test data × features, labels.
     @param features : 'numpy.ndarray' = of the dataset
     @param labels : 'numpy.ndarray' = of the dataset
     @param train_ratio : 'int' = ratio of training to testing data
     @return a tuple containing the (train_x, test_x, train_y, test_y),
        the number of testing and training examples, and the
        dimensionality
     '''
    (num_examples,) = labels.shape                  # num of valid examples
    num_test = (num_examples//(train_ratio + 1))    # num of test examples
    num_train = (num_examples - num_test)       # num of training examples
    num_dims = features.shape[1]                    # dimensionality

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
