#!/usr/bin/env python3
# encoding: utf-8

import math
import csv
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.special import expit as s
from random import shuffle
from random import randint
from random import randrange

from numpy import log as ln
from numpy import dot
from sklearn import svm
from sklearn.preprocessing import scale as normalize
from sklearn.utils import shuffle

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

# TODO fix taking log of values really close to zero by setting to .0000001
def logistic_grad_fn(w, x = train, y = labels[0]):
    ans = np.sum([(y[i] - s( dot(x[i], w) [0])) * x[i].reshape(len(x[i]),1)
                  for i in range(len(x))], axis = 0)
    return ans

def stochastic_grad_fn(w, x = train, y = labels[0]):
    # def logistic_grad_fn(w, x = train, y= labels[0]):
    i = randrange(len(x))
    ans = ((y[i] - s( dot(x[i], w) )[0]) * x[i].reshape(len(x[i]),1))
    return ans

def grad_des(epsilon = 1e-4, iterations = 100, dec_epsilon = False, x = train, y = labels[0], display = []):
    w_new = np.array([ [1] for i in range(len(x[0]))])
    assert iterations > 0

    if dec_epsilon:
        w_old = w_new
        w_new = w_old - epsilon * logistic_grad_fn(x = train, w = w_old, y = labels[0])
        for i in range(1, iterations):
            w_old = w_new
            w_new = w_old - epsilon / i **2 * logistic_grad_fn(x = train, w = w_old, y = labels[0])
            if i in display:
                print("R(w): {}, iterations : {}, epsilon: {:e}".format(R(w_new), i, epsilon))
    else:
        for i in range(iterations):
            w_old = w_new
            w_new = w_old - epsilon * logistic_grad_fn(x = train, w = w_old, y = labels[0])
            if i in display:
                print("R(w): {}, iterations : {}, epsilon: {:e}".format(R(w), i, epsilon))
            # TODO add print code if display
    return w_new

def stochastic_grad_des(epsilon = 1e-5, iterations = 100, dec_epsilon = False, display = []):
    w_new = np.array([[1] for i in range(len(train[0]))])
    assert iterations > 0
    if dec_epsilon:
        w_old = w_new
        w_new = w_old - epsilon * stochastic_grad_fn(x = train, w = w_old, y = labels[0])
        for i in range(1, iterations):
            w_old = w_new
            w_new = w_old - epsilon / i **2 * stochastic_grad_fn(x = train, w = w_old, y = labels[0])
    else:
        for _ in range(iterations):
            w_old = w_new
            w_new = w_old - epsilon * stochastic_grad_fn(x = train, w = w_old, y = labels[0])
            # TODO add print code if display
    return w_new


# TODO find out how I'm taking the log of 0 or something near it
def R(w, x = train, y = labels[0]):
    return -np.sum( y[i] * ln( np.maximum(.000000001,s(dot(x[i], w)[0]))) + (1 - y[i]) * ln( np.maximum(.000000001, 1 - s(dot(x[i],w)[0]))) for i in range(len(x)))

# def predict(x = train, w = grad_des()):
    # return np.dot(x, w)

iteration_list = [1000, 2000, 3000, 4000, 5000,]
# TODO grep for 'plt' and add a savefig

def plot_risk(epsilon = 1e-3, dec_epsilon = False, t1 = False, t2 = False, t3 = False, batch = True, show = False):
    indep = []
    dep = iteration_list
    if batch:
        f = grad_des
    else:
        f = stochastic_grad_des
    if t1:
        x_trans = normalize(train)
    elif t2:
        x_trans = log_transform_matrix(train)
    elif t3:
        x_trans = binarize_matrix(train)
    else:
        x_trans = train

    w = f(epsilon = epsilon, x = x_trans, iterations = iteration_list[-1], dec_epsilon = True, y = labels[0], display = iteration_list)
    indep.append( R(w) )
    plt.scatter(indep, dep)
    if show:
        plt.show()


def safelog(x):
    return np.log(max(x, 1e-8))

# def new risk
# def new grad
def k(x, z, rho = 1, d = 2):
    return ( dot(x.T, z) + rho)**d

def create_kernel_matrix(x = train):
    mat = np.eye(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            mat[i][j] = k(x[i], x[j])
    return mat

K = create_kernel_matrix()

def kernel_gradient_descent(x, y, epsilon=1e-5, iterations = 1000, lamb = 1e-3):
    # 1d array
    a = np.array([ [1] for i in range(len(x))])

    for m in range(iterations):
        i = randrange(len(y))
        print("iteration: {}, inital a: {}".format(m, a))
        a[i] = a[i] - (epsilon * lamb * a[i]) + epsilon * (y[i] - s(dot(K, a)[i]))
        print("iteration: {}, new a: {}".format(m, a))
        for j in range(len(y)):
            if j != i:
                a[j] = a[j] - epsilon * lamb * a[j]
        arr = [y[i] * safelog(s(a[j] * K[i,j])) + (1 - y[i]) * safelog(1 - s(a[j] * K[i,j])) for j in range(len(x))]
        if m % 100 == 0:
            risk = -np.sum(arr)
            print("risk: {}, m: {}".format(risk,m))
    print("a: {}".format(a))
    return a
# TODO find way to add R(w) and iteration number to list to plot
# Shuffle y in stochastic gradient descent to make it work? (do i need this?)
# TODO save weights you get from running computations so you can use them on training set.

def main():
    # grad_des(dec_epsilon = True)
    # stochastic_grad_des(dec_epsilon = True)
    # plot_risk(show=True)
    kernel_gradient_descent(x=train,y=labels[0], epsilon=1e-5)

main()

