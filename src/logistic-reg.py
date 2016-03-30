#!/usr/bin/env python3
# encoding: utf-8

import math
import csv
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
from sklearn.preprocessing import scale as normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# Must be run from top of project
data = sio.loadmat('./spam_dataset/spam_data.mat')
train = data['training_data']
labels = data['training_labels']

test = data['test_data']

# i) Standardize each column to have mean 0 and unit variance.

# X_scaled = normalize(train)

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

def logistic_grad_fn(x, w, y):
    """
    x: (n,d)
    w: (d,1)
    y: (n,1)
    """
    xw = np.dot(x, w)
    s = scipy.special.expit(xw)
    a = y - s
    return np.dot( (y - s), x)

# def stochastic_logistic_grad_fn():


def grad_des(precision = .001, epsilon = .1, iterations = 1000, while_precision = False): # TODO grad fn

    x_old = 
    # x_new = # TODO init a vector of all 1s here so it is of the same dim as the grad
    # TODO XXX randomly init x_new at one of the data points

    # [] TODO: find function to calculate gradient from sympy
    # [] TODO: manually plug n logistic gradient here
    grad = 0
    if iterations > 0:
        for _ in range(iterations):
            x_old = x_new
            x_new = x_old - epsilon * grad_fn(x_old)
        return x_new


    elif while_precision:
        while abs(x_new - x_old) > precision:
            x_old = x_new
            x_new = x_old - epsilon * grad_fn(x_old)
        return x_new



def stochastic_grad_des(precision = .001, x_old = 0, J, epsilon = .1):
    # Choose an initial vector of parameters w and learning rate \eta.
    # Repeat until an approximate minimum is obtained:
    # Randomly shuffle examples in the training set.
        # For \! i in 1..n:
            # \! w = w - epsilon * grad Q_i(w).


