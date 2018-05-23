# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:51:14 2018

@author: kruegkj

RSI Indicator

rsi_indicator.py
"""
import sys
sys.path.append('../lib')
import numpy as np
import retrieve_issue_data
import pandas as pd
import math

def RSI(p,lb):
    # RSI technical indicator.
    # p, the series having its RSI computed.
    # lb, the lookback period, does not need to be integer.
    #     typical values in the range of 1.5 to 5.0.
    # Return is a numpy array with values in the range 0.0 to 1.0.
    nrows = p.shape[0]
    print (nrows)
    lam = 2.0 / (lb + 1.0)
    UpMove = np.zeros(nrows)
    DnMove = np.zeros(nrows)
    UpMoveSm = np.zeros(nrows)
    DnMoveSm = np.zeros(nrows)
    Numer = np.zeros(nrows)
    Denom = np.zeros(nrows)
    pChg = np.zeros(nrows)
    RSISeries = np.zeros(nrows)
    # Compute pChg in points using a loop.
    for i in range (1,nrows):
        pChg[i] = p[i] - p[i-1]    
    # Compute pChg as a percentage using a built-in method.
#    pChg = p.pct_change()
    UpMove = np.where(pChg>0,pChg,0)
    DnMove = np.where(pChg<0,-pChg,0)
    
    for i in range(1,nrows):
        UpMoveSm[i] = lam*UpMove[i] + (1.0-lam)*UpMoveSm[i-1]
        DnMoveSm[i] = lam*DnMove[i] + (1.0-lam)*DnMoveSm[i-1]
        Numer[i] = UpMoveSm[i]
        Denom[i] = UpMoveSm[i] + DnMoveSm[i]
        if Denom[i] <= 0:
            RSISeries[i] = 0.5
        else:
            RSISeries[i] =  Numer[i]/Denom[i]
    return(RSISeries)
    
if __name__ == "__main__":
    issue = "xly"
    lookback = 2.3
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"  
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    dataSet['Pri_RSI'] = RSI(dataSet.Pri, lookback)
    
    # Plot price and belong indicator
    rsiDataSet = dataSet.ix[startDate:endDate]
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                       xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                       ylim=(0,1))
    x = np.linspace(0, 10)
    ax1.plot(rsiDataSet['Pri'])
    ax2.plot(rsiDataSet['Pri_RSI']);
    
    