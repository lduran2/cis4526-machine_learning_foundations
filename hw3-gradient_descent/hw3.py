#!/usr/bin/env python3
r'''
 ./hw3.py
 Implements and tests a gradient descent algorithm with numerical
 differentiation.

 By        : Leomar Durán <https://github.com/lduran2/>
 When      : 2021-11-05t15:44
 Where     : Temple University
 For       : CIS 4526
 Version   : 1.0.1
 Dataset   : https://archive.ics.uci.edu/ml/datasets/Wine
 Canonical : https://github.com/lduran2/cis-4526-hw2-kNN-vs-pocket/blob/master/hw2.py

 CHANGELOG :
    v1.0.1 - 2021-11-05t15:44
        named the features

    v1.0.0 - 2021-11-05t15:29
        printing the rows of the data file
 '''
#

from csv import DictReader

DATA_FILENAME = r'wine.data'  # name of file holding the dataset

FEATURES = (  # tuple of the names of features
    r'Alcohol',
    r'Malic_acid',
    r'Ash',
    r'Alcalinity_of_ash',
    r'Magnesium',
    r'Total_phenols',
    r'Flavanoids',
    r'Nonflavanoid_phenols',
    r'Proanthocyanins',
    r'Color_intensity',
    r'Hue',
    r'OD280__OD315_of_diluted_wines',
    r'Proline'
) # end FEATURES

def main():
    r'''
     classify the wine samples
     '''
    # open the CSV file
    with open(DATA_FILENAME) as datafile:
        # place a dictionary reader on the dataset
        reader = DictReader(datafile, fieldnames=FEATURES)

        # loop through the rows
        for row in reader:
            print(row)
        # end for row in reader
    # end with open(DATA_FILENAME) as csvfile
# end def main()

# run main if this package was run
if (__name__==r'__main__'):
    main()
# end if (__name__==r'__main__')
