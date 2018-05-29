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
        # if % change is 0, change to small number
        if (abs(g[i]) < 0.0001):
            g[i] = 0.0001
    return g
    
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
    issue = "xle"
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
    
    testFirstYear = "2015-10-01"
    testFinalYear = "2016-01-30"
    qtPlot = targetDataSet.ix[testFirstYear:testFinalYear]
    
    numSubPlots = 2
    # format the ticks
    fig, axes = plt.subplots(numSubPlots,1, figsize=(numSubPlots*5,8), sharex=True)
    
    axes[0].plot(qtPlot['Close'], label=issue)
    axes[1].plot(qtPlot['beLong'], label='beLong');

    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0.1)
    
    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axes:
            ax.label_outer()
            ax.legend(loc='upper left', frameon=True, fontsize=8)
            ax.grid(True, which='both')
            fig.autofmt_xdate()
            ax.xaxis_date()
            ax.autoscale_view()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            ax.minorticks_on()
            ax.tick_params(axis='y',which='minor',bottom='off')
    
