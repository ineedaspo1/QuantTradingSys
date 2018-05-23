# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:51:14 2018

@author: kruegkj

Z Score Transform

zScore_transform.py
"""
import sys
sys.path.append('../lib')
import numpy as np
import retrieve_issue_data
import pandas as pd
import math

def zScore(p,lb):
    # z score statistic.
    # p, the series having its z-score computed.
    # lb, the lookback period, an integer.
    #     the length used for the average and standard deviation.
    #     typical values 3 to 10.
    # Return is a numpy array with values as z-scores centered on 0.0.
    nrows = p.shape[0]
    st = np.zeros(nrows)
    ma = np.zeros(nrows)
    # use the pandas sliding window functions.
    st = pd.rolling_std(p,lb)
    ma = pd.rolling_mean(p,lb)
    z = np.zeros(nrows)
    for i in range(lb,nrows):
        z[i] = (p[i]-ma[i])/st[i]
    return z
    
if __name__ == "__main__":
    issue = "xly"
    lookback = 16
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"  
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    dataSet['Pri_zScr'] = zScore(dataSet.Pri, lookback)
    
    # Plot price and belong indicator
    zScrDataSet = dataSet.ix[startDate:endDate]
    fig = plt.figure(figsize=(15,8  ))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
                       
    x = np.linspace(0, 10)
    ax1.plot(zScrDataSet['Pri'])
    # True range is computed as a fraction of the closing price.
    ax2.plot(zScrDataSet['Pri_zScr']);
    
    