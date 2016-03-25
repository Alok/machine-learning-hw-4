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

# [] TODO add bias term by appending 1 to beginning of every feature and to the
# list of labels

housing_data = sio.loadmat('../housing_dataset/housing_data.mat')


x_train    = housing_data['Xtrain']
y_train    = housing_data['Ytrain']

x_validate = housing_data['Xvalidate']
y_validate = housing_data['Yvalidate']

def ridge_regression_model(X = x_train, y = y_train, epsilon = .001):
    """
    TODO: Returns a set of weights for a ridge regression model.

    :X: (n,m) design matrix
    :y: (n,1) labels
    :epsilon: hyperparameter
    :returns: a (m,1) vector of weights

    """
    # get number of columns for the num of rows of 'w'
    N = X.shape[1]
    X_t = np.transpose(X)

    I = np.identity(N)

    A = np.linalg.inv( np.dot(X_t, X) + epsilon * I )
    B = np.dot(A, X_t)
    C = np.dot(B, y)

    return C

# [] TODO make fn that returns a validation set and training sets for k-fold
# cross validation

def k_fold(k=10, sample_size = 10000, data = x_train, labels = y_train, fold_num = 0):
    """
    TODO: implement k-fold cross validation

    :x: TODO
    :returns: ((set of {left + right}, current), (y {left right}, y current))

    """
    s = sample_size // k
    i = fold_num

    left_x     = data[:i * s ]
    current_x  = data[i * s : (i+1) * s]
    right_x    = data[(i+1) * s:]

    left_y    = labels[ : i * s ]
    current_y = labels[i * s : (i+1) * s]
    right_y   = labels[(i+1) * s :]

    x = np.concatenate((left_x, right_x))
    y = np.concatenate((left_y, right_y))

        return ((x, current_x), (y, current_y))

        # y = np.concatenate((left_y, right_y)).ravel()

# TODO train on each cross validation set and find lowest error rate with diff hyperparams

hyperparam_vals = [1e-10,1.5e-7,1e-5,1e-3,1e-1,1e0,1e1,1e2,1e3,1e4]

folds = 10

for param in hyperparam_vals:
    for n in range(folds):
        x_cross = k_fold(k = folds, fold_num = n)[0][0]
        y_cross = k_fold(k = folds, fold_num = n)[1][0]
    w = ridge_regression_model
