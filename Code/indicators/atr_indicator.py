# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:51:14 2018

@author: kruegkj

ATR Indicator

atr_indicator.py
"""
import sys
sys.path.append('../lib')
import numpy as np
import retrieve_issue_data
import pandas as pd
import math

def ATR(ph,pl,pc,lb):
    # Average True Range technical indicator.
    # ph, pl, pc are the series high, low, and close.
    # lb, the lookback period.  An integer number of bars.
    # True range is computed as a fraction of the closing price.
    # Return is a numpy array of floating point values.
    # Values are non-negative, with a minimum of 0.0.
    # An ATR of 5 points on a issue closing at 50 is
    #    reported as 0.10. 
    nrows = pc.shape[0]
    th = np.zeros(nrows)
    tl = np.zeros(nrows)
    tc = np.zeros(nrows)
    tr = np.zeros(nrows)
    trAvg = np.zeros(nrows)
    
    for i in range(1,nrows):
        if ph[i] > pc[i-1]:
            th[i] = ph[i]
        else:
            th[i] = pc[i-1]
        if pl[i] < pc[i-1]:
            tl[i] = pl[i]
        else:
            tl[i] = pc[i-1]
        tr[i] = th[i] - tl[i]
    for i in range(lb,nrows):
        trAvg[i] = tr[i]            
        for j in range(1,lb-1):
            trAvg[i] = trAvg[i] + tr[i-j]
        trAvg[i] = trAvg[i] / lb
        trAvg[i] = trAvg[i] / pc[i]    
    return(trAvg)
    
if __name__ == "__main__":
    issue = "xly"
    lookback = 5
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"  
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    dataSet['Pri_ATR'] = ATR(dataSet.High,dataSet.Low,dataSet.Pri, lookback)
    
    # Plot price and belong indicator
    atrDataSet = dataSet.ix[startDate:endDate]
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                       xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
                       
    x = np.linspace(0, 10)
    ax1.plot(atrDataSet['Pri'])
    # True range is computed as a fraction of the closing price.
    ax2.plot(atrDataSet['Pri_ATR']);
    
    