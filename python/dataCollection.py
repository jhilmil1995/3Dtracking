import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def collectData():
    #ser = Serial.serial
    pass

def get_indicators(vec):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    mean = np.mean(vec)
    slope = calc_slope(np.arange(len(vec)),vec)
    std = np.std(vec)
    mx = np.max(vec)
    mi = np.min(vec)
    sdev = np.std(vec)
    return mean, slope, std, mx, mi, sdev

def calc_slope(x,y):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = np.sum(np.abs(np.subtract(x,x_mean)))
    y_dev = np.sum(np.abs(np.subtract(y,y_mean)))

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope



