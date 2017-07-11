#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def show_error(error_train, error_test, iteration, unit = 1):
    plt.figure("Error Performance")
    x = np.array([x*unit for x in range(iteration//unit)])
    y = np.array([error_train[y*unit] for y in range(iteration//unit)])
    
    plt.semilogy(x, y, linewidth=1, color = "red", label = "Train Error")
    plt.ylim(np.min(y)/10,np.max(y))
    #plt.plot(x, y, 'r')
    if error_test != None:
        x2 = np.array([x*unit for x in range(iteration//unit)])
        y2 = np.array([error_test[y*unit] for y in range(iteration//unit)])
        plt.semilogy(x2, y2, linewidth=1, color = "green", label = "Test Error")
        plt.ylim(np.min(y+y2)/10,np.max(y+y2))
    
    plt.xlim(0, iteration)
    #plt.axis([0, iteration, 0, np.max(y)])
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()

def show_gradient(gradient, iteration, unit = 1):
    plt.figure("Gradient")
    x = np.array([x*unit for x in range(iteration//unit)])
    y = np.array([gradient[y*unit] for y in range(iteration//unit)])
    plt.semilogy(x, y, linewidth=1, color = "blue", label = "Gadient")
    plt.xlim(0, iteration)
    plt.ylim(np.min(y)/10,np.max(y))
    plt.xlabel("Iteration")
    plt.ylabel("Gradient")
    plt.legend()


def show_regression(ouputs_np, targets_np, k, b, r, title = "Regression"):
    ouputs = ouputs_np.reshape(-1,1)
    targets = targets_np.reshape(-1,1)
    plt.figure(title)
    plt.title("R = %s" % r)
    print(len(ouputs), len(targets))
    plt.plot(targets, ouputs, '.', label = "Data")
    
    _min = np.min([np.min(ouputs) , np.min(targets)])
    _max = np.max([np.max(ouputs) , np.max(targets)])
    
    plt.axis([_min, _max, _min, _max])
    x = np.linspace(_min, _max, len(targets))
    plt.plot(x, x, '--', label = "Y = T")
    
    plt.plot(x, k * x + b, 'g', label = "Fit")
    plt.xlabel("Targets")
    plt.ylabel("Outputs = %s * Targets + %s" % (k, b))
    plt.legend()
