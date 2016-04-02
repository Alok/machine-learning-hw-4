#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from numpy import log as ln
import sklearn.preprocessing as skp
import scipy.io as sio
from ptpdb import set_trace
import random
import math
from sklearn.utils import shuffle

def tr_standardize(data):
    return np.array([skp.scale(x) for x in data])

def tr_log(data):
    return np.array([[ln(e + 0.1) for e in x] for x in data])

def tr_binarize(data):
    def indicator(e):
        if e > 0:
            return 1
        return 0
    return np.array([[indicator(e) for e in x] for x in data])

def logistic_reg(option, epsilon, st, kern=False):
    data = sio.loadmat('spam_dataset/spam_data.mat')
    X = data['training_data']
    set_trace()
    y = data['training_labels'].T
    set_trace()

    if option == 0:
        X = tr_standardize(X)
    elif option == 1:
        X = tr_log(X)
    elif option == 2:
        X = tr_binarize(X)

    X, y = shuffle(X, y)

    X, Xv = X[:3418], X[3418:5172]
    y, yv = y[:3418], y[3418:5172]

    if kern:
        a = kernel_gradient_descent(X, y, epsilon)
        w = np.dot(X.T, a)
    else:
        w = gradient_descent(X, y, epsilon, stochastic=st)
    def validate():
        predictions = []
        for x in Xv:
            h = np.dot(x, w)
            if sigmoid(h) > 0.5:
                y_hat = 1
            else:
                y_hat = 0
            predictions.append(y_hat)
        set_trace()
        score = 0
        for i, p in enumerate(predictions):
            if predictions[i] == yv[i]:
                score += 1
        print("score is %.2f"%(score/len(yv)))
    validate()

def gradient_descent(X, y, epsilon, stochastic=False):
    w = np.ones(32)
    arr = [y[i] * safelog(sigmoid(np.dot(w.T, X[i]))) \
        + (1 - y[i]) * safelog(1 - sigmoid(np.dot(w.T, X[i]))) \
        for i in range(len(X))]
    R = -sum(arr)
    count = 0
    precision = 1e-5
    deltaR = math.inf
    if stochastic:
        max_counts = 5000
    else:
        max_counts = 1000
    while deltaR > precision and count < max_counts:
        if stochastic:
            i = random.randrange(0, len(X))
            descent = y[i] - sigmoid(np.dot(X[i].T, w)) * X[i]
        else:
            descent = sum([(y[i] - sigmoid(np.dot(X[i].T, w))) * (X[i]) for i in range(len(X))])
        w += epsilon * descent
        arr = [y[i] * safelog(sigmoid(np.dot(w.T, X[i]))) + (1 - y[i]) * \
         safelog(1 - sigmoid(np.dot(w.T, X[i]))) for i in range(len(X))]
        deltaR = abs(R + sum(arr))
        R = -sum(arr)
        count += 1
        if count % 100 == 0:
            print(R, count)
    return w

def sigmoid(gamma):
    return 1 / (1 + np.exp(-gamma) + 1e-8)

def safelog(x):
    return np.log(max(x, 1e-8))

def kernel_gradient_descent(X, y, epsilon):
    n = len(X)
    a = np.array(np.ones(n), ndmin=2).T
    lamb = 1e-3
    rho = 12
    count = 0

    def calculate_kernel_matrix():
        kernel_matrix = [[0 for _ in range(n)] for __ in range(n)]
        for i in range(n):
            if i % (n // 10) == 0:
                print("calculating kernel matrix...")
            for j in range(n):
                kernel_matrix[i][j] = kernel(X[i], X[j], rho)
        return np.matrix(kernel_matrix)

    kernel_matrix = calculate_kernel_matrix()

    R = math.inf

    while R > 0:
        i = random.randrange(0, n)
        a[i] = a[i] - (lamb * a[i]) + epsilon * (y[i] - sigmoid(np.dot(kernel_matrix[i], a)))
        for h in range(n):
            if h != i:
                a[h] = a[h] - lamb * a[h]

        inside_sigmoid = sum([a[j] * kernel_matrix[j, i] for j in range(n)])

        R = -sum([y[j] * safelog(sigmoid(inside_sigmoid))\
            + (1 - y[j]) * safelog(1 - sigmoid(inside_sigmoid))\
            for j in range(n)]) + lamb * np.dot(a.T, a)
        count += 1
        if count % 100 == 0:
            print(R, count)
        if count == 1000:
            break
    return a

def kernel_gradient_descent_TEST(X, y, epsilon):
    X = X[:500]
    y = y[:500]

    n = len(X)
    a = np.array(np.ones(n), ndmin=2).T
    lamb = 1e-3
    rho = 12
    count = 0

    def calculate_kernel_matrix():
        kernel_matrix = [[0 for _ in range(n)] for __ in range(n)]
        for i in range(n):
            if i % (n // 10) == 0:
                print("calculating kernel matrix...")
            for j in range(n):
                kernel_matrix[i][j] = kernel(X[i], X[j], rho)
        return np.matrix(kernel_matrix)

    kernel_matrix = calculate_kernel_matrix()
    set_trace()

    R = math.inf

    set_trace()
    while R > 0:
        i = random.randrange(0, n)
        a[i] = a[i] - (lamb * a[i]) + epsilon * (y[i] - sigmoid(np.dot(kernel_matrix[i], a)))
        for h in range(n):
            if h != i:
                a[h] = a[h] - lamb * a[h]

        inside_sigmoid = sum([a[j] * kernel_matrix[j, i] for j in range(n)])

        R = -sum([y[j] * safelog(sigmoid(inside_sigmoid))\
            + (1 - y[j]) * safelog(1 - sigmoid(inside_sigmoid))\
            for j in range(n)]) + lamb * np.dot(a.T, a)
        count += 1
        if count % 100 == 0:
            print(R, count)
        if count == 10000:
            break
    return a


def kernel(x, z, rho, degree=1):
    return (np.dot(x.T, z) + rho) ** degree

if __name__ == "__main__":
    import sys
    option = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    stochastic = (sys.argv[3] == "stochastic")
    kern = (sys.argv[4] == "kernel")
    logistic_reg(option, epsilon, stochastic, kern)
