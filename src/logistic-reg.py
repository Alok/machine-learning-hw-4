#!/usr/bin/env python3
# encoding: utf-8

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

spam_data = sio.loadmat('../spam_dataset/spam_data.mat')
spam_train = spam_data['training_data']
spam_test = spam_data['test_data']

print("spam_train[1]: {}".format(len(spam_train[1])))
# i) Standardize each column to have mean 0 and unit variance.

def mean_of_vecs(mat):
    """
    m = [
            v1
            v2
            v3
        ]

    where v_i is a row vector.

    :returns: (1,n) vector

    """
    def mean(m, index):
        """
        Find mean at a certain index of a set of vectors arranged as above.

        :returns: float
        """
        v = [mat[i][index] for i in range(len(mat))]
        return sum v / len(v)

    return = [mean(mat, j) for j in range(len(mat))]

def center_data(mat):
    """
    TODO: Docstring for center_data

    :mat: matrix whose columns need to be aligned
    :returns: TODO

    """
    x_bar = mean_of_vecs(mat)

    # [] TODO: subtract each row from mean

def log_transform_matrix(mat):
    def log_transform_list(lst):
        assert all(x > -.1 for x in lst)
        f = lambda x: math.log( x + 0.1)
        return [f(i) for i in lst]

    return np.array([log_transform_list(lst) for lst in mat])



# ii) Transform the features using Xij := log(Xij +0.1), where the Xij’s are the entries of the design matrix.
def binarize_matrix(mat):
    def binarize_list(lst):
        f = lambda x: 1 if x > 0 else 0
        return [f(i) for i in lst]

    return np.array([binarize_list(lst) for lst in mat])
