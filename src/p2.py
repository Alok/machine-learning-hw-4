#!/usr/bin/env python3
# encoding: utf-8

from math import log as ln
import os
import scipy
from scipy.special import expit as s
import numpy as np
from numpy import dot
from pprint import pprint

# s = scipy.special.expit
x = np.array([ [0, 3,1], [1,3,1], [0,1,1], [1,1,1] ])

y = np.array([[1], [1], [0], [0]])

w0 = np.array([[-2] ,[1] ,[0]])


def logistic_grad_fn(w):
    ans = np.sum([(y[i] - s( dot(x[i], w) )[0]) *
                  x[i].reshape(len(x[i]),1) for i in
                  range(len(x))], axis = 0)
    return ans

def grad_des(iterations = 1, w = w0):
    assert iterations > 0
    w_new = w
    for _ in range(iterations):
        w_old = w_new
        w_new = w_old + logistic_grad_fn(w_old)
    return w_new

def R(w):
    return -np.sum([ y[i] * ln(s(dot(x[i],w)[0])) +
                     (1- y[i]) * ln(1 - s( dot(x[i],w)[0]))
                     for i in range(len(x))])

def mu(i):
    w1 = grad_des(1, w0)
    if i ==0:
        t = w0
    elif i ==1:
        t = w1
    return [s(np.dot(t.T, x[i].reshape(3,1))[0]) for i in range(4)]

def main():
    pprint("R(w0): {}".format(R(w0)))
    w1 = grad_des(1, w0)
    pprint("w1: {}".format(w1))
    w2 = grad_des(1, w1)
    pprint("R(w1): {}".format(R(w1)))
    pprint("mu(0): {}".format( mu(0)))
    pprint("w2: {}".format(w2))
    pprint("mu(1): {}".format( mu(1)))
    pprint("R(w2): {}".format(R(w2)))

main()
