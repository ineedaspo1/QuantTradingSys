# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:51:14 2018

@author: kruegkj

ROC Indicator

roc_indicator.py
"""
import sys
sys.path.append('../lib')
import numpy as np
import retrieve_issue_data
import pandas as pd
import math

def ROC(p,lb):
    # Rate of change technical indicator.
    # p, the series having its ROC computed.
    # lb, the lookback period.  Typically 1.
    # Return is a numpy array with values as decimal fractions.
    # A 1% change is 0.01.
    nrows = p.shape[0]
    r = np.zeros(nrows)
    for i in range(lb, nrows):
        r[i] = (p[i]-p[i-lb])/p[i-lb]
    return(r)
    
if __name__ == "__main__":
    issue = "xly"
    lookback = 16
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"  
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    dataSet['Pri_ROC'] = ROC(dataSet.Pri, lookback)
    
    # Plot price and belong indicator
    rocDataSet = dataSet.ix[startDate:endDate]
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
                       
    x = np.linspace(0, 10)
    ax1.plot(rocDataSet['Pri'])
    # True range is computed as a fraction of the closing price.
    ax2.plot(rocDataSet['Pri_ROC']);
    
    