# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:31:00 2018

@author: kruegkj

compute_target.py

Compute target for price series based on gain/loss over
threshold level. Target price is +1 is over/under threshold for long/short

Inputs:
    Issue dataframe
    Direction (long or short)
    Threshold
    
Output:
    Dataframe with gainAhead, beLong

Future addition: Short
"""
import numpy as np
from retrieve_issue_data import *
import matplotlib.pyplot as plt

def gainAhead(p):
    # Computes change in the next 1 bar.
    # p, the base series.
    # Return is a numpy array of changes.
    # A change of 1% is 0.01
    # The final value is unknown.  Its value is 0.0.
    nrows = p.shape[0]
    g = np.zeros(nrows)
    for i in range(0,nrows-1):
        g[i] = (p[i+1]-p[i])/p[i]
    return(g)
    
def priceChange(p):
    nrows = p.shape[0]
    pc = np.zeros(nrows)
    for i in range(1,nrows):
        pc[i] = (p[i]-p[i-1])/p[i-1]
    return pc

def setTarget(p, direction, beLongThreshold):
    p['gainAhead'] = gainAhead(p['Pri'])
    p['beLong'] = np.where(p.gainAhead>beLongThreshold,1,-1)
    return p

if __name__ == "__main__":
    issue = "XRT"
    dataLoadStartDate = "1998-12-22"
    dataLoadEndDate = "2016-01-04"
    beLongThreshold = 0.01
    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    
    targetDataSet = setTarget(dataSet, "Long", beLongThreshold)
    nrows = targetDataSet.shape[0]
    print ("nrows: ", nrows)
    print (targetDataSet.shape)
    print (targetDataSet.tail(10))
    
    print ("beLong counts: ")
    print (targetDataSet['beLong'].value_counts())
    print ("out of ", nrows)
    
    testFirstYear = "2015-01-01"
    testFinalYear = "2015-06-30"
    qtPlot = targetDataSet.ix[testFirstYear:testFinalYear]
    
    # Plot price and belong indicator
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                       xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                       ylim=(-1,1))
    x = np.linspace(0, 10)
    ax1.plot(qtPlot['Close'])
    ax2.plot(qtPlot['beLong']);
    
