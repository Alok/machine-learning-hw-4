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

# x_train = normalize(x_train)
# x_validate = normalize(x_validate)

# ============= FUNCTIONS =============

# residual sum of squares error
def rss(pred_labels, true_labels):
    errors = pred_labels - true_labels
    err_sq = errors**2
    return np.sum(err_sq)

def add_bias(vec, bias_term = 1):
    return np.insert(vec, [len(vec[0])], [[bias_term] for i in range(len(vec))], axis = 1)


# ============= RIDGE REGRESSION =============
def ridge_regression_model(X = x_train, y = y_train, epsilon = 1e-4):
    """
    Returns a set of weights for a ridge regression model.

    :X: (n, m) design matrix
    :y: (n, 1) labels
    :epsilon: learning rate
    :returns: a (m, 1) vector of weights

    """

    # get number of columns for the num of rows of 'w'
    N = X.shape[1]
    X_t = X.T

    I = np.identity(N)

    A = np.linalg.inv( np.dot(X_t, X) + epsilon * I )
    B = np.dot(A, X_t)
    C = np.dot(B, y)

    return C

def predict(x = x_validate, w = ridge_regression_model()):
    return np.dot(x, w)


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
# TODO plot the value of each coefficient against the index of the coefficient
def test_k_fold(param_list = hyperparam_vals, data = x_train, labels = y_train, folds = 10):

    for param in hyperparam_vals:
        for n in range(folds):
            x_valid = k_fold( data = x_train , fold_num = n, k = folds )[0]
            x_cross = k_fold( data = x_train , fold_num = n, k = folds )[1]

            y_valid = k_fold( data = y_train , fold_num = n, k = folds )[0]
            y_cross = k_fold( data = y_train , fold_num = n, k = folds )[1]

            w = ridge_regression_model(epsilon = param)
            y_pred = np.dot(x_cross, w)
            print (w, y_pred)


def plot_w(w, show = False):
    indep = [i for i in range(len(w))]
    dep = [w[i] for i in range(len(w))]
    plt.scatter(indep, dep)
    if show:
        plt.show()

def main():
    print (ridge_regression_model().shape)
    print (ridge_regression_model())
    print("rss(): {:}".format(rss(predict(), y_validate)))
    plot_w(ridge_regression_model(), show=True)

main()
