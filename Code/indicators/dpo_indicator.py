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

def DPO(p,lb):
    # Detrended price oscillator. 
    # A high pass filter.
    # p, the series being transformed.
    # lb, the lookback period, a real number.
    # Uses pandas ewma function.
    # Return is a numpy array with values centered on 0.0.
    nrows = p.shape[0]
    ma = pd.ewma(p,span=lb)
    d = np.zeros(nrows)
    for i in range(1,nrows):
        d[i] = (p[i]-ma[i])/ma[i]
    return(d)
    
if __name__ == "__main__":
    issue = "xly"
    lookback = 5
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"  
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    dataSet['Pri_DPO'] = DPO(dataSet.Pri, lookback)
    
    # Plot price and belong indicator
    dpoDataSet = dataSet.ix[startDate:endDate]
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
                       
    x = np.linspace(0, 10)
    ax1.plot(dpoDataSet['Pri'])
    # True range is computed as a fraction of the closing price.
    ax2.plot(dpoDataSet['Pri_DPO']);
    
    