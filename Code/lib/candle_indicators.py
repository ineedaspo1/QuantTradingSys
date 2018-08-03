# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

candle_indicators.py
"""

from retrieve_data import *


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class CandleIndicators:
      
    def HigherClose(self,p,num_days,feature_dict):
        # Returns true if closing price greater than closeing price num_days previous
        nrows = p.shape[0]

        return d, feature_dict
    
    
#    dataSet['1DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag1']
#    dataSet['2DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag2']
#    dataSet['3DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag3']
#    dataSet['4DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag4']
#    
#    dataSet['1DayLowerClose'] = dataSet['Pri'] < dataSet['Pri_lag1']
#    dataSet['2DayLowerClose'] = dataSet['Pri'] < dataSet['Pri_lag2']
#    dataSet['3DayLowerClose'] = dataSet['Pri'] < dataSet['Pri_lag3']
#    dataSet['4DayLowerClose'] = dataSet['Pri'] < dataSet['Pri_lag4']
    
    
if __name__ == "__main__":
    from plot_utils import *
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    plotIt = PlotUtility()
    
#    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
#    plotIt.plot_v1(dataSet['Pri'], plotTitle)
    dataSet.CloseHigher=[]
    num_days = 3
    nrows = dataSet.shape[0]
    print(nrows)
    
    for i in range (num_days+1,nrows):
        dataSet.CloseHigher[i] = dataSet.Pri[i] > dataSet.Pri[i-num_days]    
    
#    beLongThreshold = 0
#    cT = ComputeTarget()
#    mmData = cT.setTarget(dataSet, "Long", beLongThreshold)
#    
#    
#    startDate = "2015-02-01"
#    endDate = "2015-06-30"
#    rsiDataSet = dataSet.ix[startDate:endDate]
#    #fig = plt.figure(figsize=(15,8  ))
#    fig, axes = plt.subplots(5,1, figsize=(15,8), sharex=True)
#
#    axes[0].plot(rsiDataSet['Pri'], label=issue)
#    axes[1].plot(rsiDataSet['RSI'], label='RSI');
#    axes[2].plot(rsiDataSet['ROC'], label='ROC');
#    axes[3].plot(rsiDataSet['DPO'], label='DPO');
#    axes[4].plot(rsiDataSet['ATR'], label='ATR');
#    
#    # Bring subplots close to each other.
#    plt.subplots_adjust(hspace=0)
#    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
#    # Hide x labels and tick labels for all but bottom plot.
#    for ax in axes:
#        ax.label_outer()
#        ax.legend(loc='upper left', frameon=False)
#    
    
