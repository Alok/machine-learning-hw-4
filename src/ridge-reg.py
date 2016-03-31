#!/usr/bin/env python3
# encoding: utf-8

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import scale as normalize
from sklearn.metrics import confusion_matrix
from numpy.linalg import norm
from numpy import array as arr

# ============= CONSTANTS =============

hyperparam_vals = [1e-10, 1.5e-7, 1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

# Must be run from top of project
housing_data = sio.loadmat('./housing_dataset/housing_data.mat')

x_train    = housing_data['Xtrain']
y_train    = housing_data['Ytrain']

x_validate = housing_data['Xvalidate']
y_validate = housing_data['Yvalidate']

# ============= FUNCTIONS =============

# residual sum of squares error
def rss(pred_labels, true_labels):
    errors = pred_labels - true_labels
    err_sq = errors**2
    return sum(err_sq)

def add_bias(vec, bias_term = 1):
    return np.insert(vec, [len(vec[0])], [[bias_term] for i in range(len(vec))], axis = 1)

def center(mat):
    """
    Centers each vector in a matrix without rescaling their lengths.
    """
    return normalize(mat, with_mean = False)

# [] TODO: Normalize x
# [] TODO: get RSS of diff vector

# ============= RIDGE REGRESSION =============
def ridge_regression_model(X = x_train, y = y_train, epsilon = .001):
    """
    Returns a set of weights for a ridge regression model.

    :X: (n, m) design matrix
    :y: (n, 1) labels
    :epsilon: learning rate
    :returns: a (m, 1) vector of weights

    """

    # get number of columns for the num of rows of 'w'
    N = X.shape[1]
    X_t = np.transpose(X)

    I = np.identity(N)

    A = np.linalg.inv( np.dot(X_t, X) + epsilon * I )
    B = np.dot(A, X_t)
    C = np.dot(B, y)

    return C

# x = x_train
# y = y_train

x = normalize(x_train, with_mean = False)
y = normalize(y_train, with_mean = False)

def test_params(data, labels, param_list = hyperparam_vals):
    x = data
    y = labels

    for epsilon in param_list:
        w = ridge_regression_model(x, y, epsilon)
        xw = np.dot(x, w)
        diff = xw - y
        print("epsilon: {}, norm: {}".format(epsilon, norm(diff)))


def k_fold(data, k = 10, fold_num = 0):
    """
    k-fold cross validation.

    :returns: (current fold , set of {left + right})

    """
    s = len(data) // k
    i = fold_num

    left     = data[:i * s ]
    current  = data[i * s : (i+1) * s]
    right    = data[(i+1) * s:]

    validation_set = current

    training_set = np.concatenate((left, right))

    return (validation_set, training_set)

# TODO train on each cross validation set and find lowest error rate with diff hyperparams

def test_k_fold(param_list = hyperparam_vals, data = x_train, labels = y_train, folds = 10):

    for param in hyperparam_vals:
        for n in range(folds):
            x_valid = k_fold( data = x_train , fold_num = n, k = folds )[0]
            x_cross = k_fold( data = x_train , fold_num = n, k = folds )[1]

            y_valid = k_fold( data = y_train , fold_num = n, k = folds )[0]
            y_cross = k_fold( data = y_train , fold_num = n, k = folds )[1]

            w = ridge_regression_model()
            y_pred = np.dot(x_cross, w)
            print (w, y_pred)
