#!/usr/bin/env python3
# encoding: utf-8

import math
import csv
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from numpy import dot
from sklearn import svm
from sklearn.preprocessing import scale as normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


# Must be run from top of project
data = sio.loadmat('./spam_dataset/spam_data.mat')
train = data['training_data']
train = normalize(train)

labels = data['training_labels']

test = data['test_data']

# i) Standardize each column to have mean 0 and unit variance.


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
    s = scipy.special.expit
    # ans = np.sum([(y[i] - s( dot(x[i], w) )) * x[i].reshape(len(x[i]),1) for i in range(len(x[0]))], axis = 0)
    ans = np.sum([(y[i] - s( dot(x[i], w) )) * x[i] for i in range(len(x[0]))], axis = 0)
    return ans


"""
x: (n,d)
w: (d,1)
y: (n,1)
"""

def grad_des(epsilon = .1, iterations = 1000) : # TODO grad fn
    """
    :: array -> Int -> Int -> weight
    """

    w_new = np.array([ [1] for i in range(len(train[0]))])
    print("old: {}".format(w_new.shape))

    # x_new = # TODO init a vector of all 1s here so it is of the same dim as the grad
    # TODO XXX randomly init x_new at one of the data points

    # [] TODO: find function to calculate gradient from sympy
    # [] TODO: manually plug n logistic gradient here

    assert iterations > 0

    for _ in range(iterations):
        w_old = w_new
        w_new = w_old - epsilon * logistic_grad_fn(train, w_old, labels) # grad_fn(w_old)
    print("new: {}".format(w_new.shape))
    return w_new


def test_grad_desc(x=train, epsilon = .1, iteration=1000, y=labels):

def stochastic_grad_des(epsilon = .1, iterations = 1000) : # TODO grad fn
w_init =
    # Choose an initial vector of parameters w and learning rate \eta.
    # Repeat until an approximate minimum is obtained:
    # Randomly shuffle examples in the training set.
        # For \! i in 1..n:
            # \! w = w - epsilon * grad Q_i(w).


def plot_grad_desc(iterations, risk):
    """
    generate a single point to be plotted
    """

def decreasing_learning_rate(x):

