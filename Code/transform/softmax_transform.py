# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:51:14 2018

@author: kruegkj

SoftMax Transform

softmax_transform.py
"""
from __future__ import division
import sys
sys.path.append('../lib')
import numpy as np
import retrieve_issue_data
import math


def softmax(p,lb,lam):
    # softmax transformation.
    # p, the series being transformed.
    # lb, the lookback period, an integer.
    #     the length used for the average and standard deviation.
    #     typical values 20 to 252.  Be aware of ramp-up requirement.
    # lam, the length of the linear section.
    #     in standard deviations.
    #     typical value is 6.
    # Return is a numpy array with values in the range 0.0 to 1.0.
    nrows = p.shape[0]
    a = np.zeros(nrows)
    ma = np.zeros(nrows)
    sd = np.zeros(nrows)    
    sm = np.zeros(nrows)
    sq = np.zeros(nrows)
    y = np.zeros(nrows)
    for i in range(lb,nrows):
        sm[i] = sm[i]+p[i]
    ma[i] = sm[i] / lb
    for i in range(lb,nrows):
        sq[i] = (p[i]-ma[i])*(p[i]-ma[i])
    sd[i] = math.sqrt(sq[i]/(nrows-1))
    for i in range(lb,nrows):
        a[i] = (p[i]-ma[i])/((lam*sd[i])/(2.0*math.pi))
        y[i] = 1.0 / (1.0 + math.e**a[i])
    return y
    
if __name__ == "__main__":
    issue = "xly"
    lookback = 40
    lam = 6
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"  
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    dataSet['Pri_sftMax'] = softmax(dataSet.Pri, lookback, lam)
    print(sum(dataSet['Pri_sftMax']))
    
    # Plot price and belong indicator
    sftmaxDataSet = dataSet.ix[startDate:endDate]
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
                       
    x = np.linspace(0, 10)
    ax1.plot(sftmaxDataSet['Pri'])
    # True range is computed as a fraction of the closing price.
    ax2.plot(sftmaxDataSet['Pri_sftMax']);
    
    