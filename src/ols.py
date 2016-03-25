#!/usr/bin/env python
# encoding: utf-8

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices


# [] TODO load data and play around with it
# [] TODO add bias term by appending 1 to beginning of every feature and to the list of labels

housing_data = sio.loadmat('../housing_dataset/housing_data.mat')
spam_data = sio.loadmat('../spam_dataset/spam_data.mat')


x_validate = housing_data['Xvalidate']
y_validate = housing_data['Yvalidate']
x_train    = housing_data['Xtrain']
y_train    = housing_data['Ytrain']

spam_train = spam_data['training_data']
spam_test = spam_data['test_data']
print (y_train[1])
