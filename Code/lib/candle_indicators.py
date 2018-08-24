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
      
    def HigherClose(self,dataSet,num_days,feature_dict):
        # Returns true if closing price greater than closeing price num_days previous
        column_name = str(num_days) + 'dHigherCls'
        nrows = dataSet.shape[0]
        pChg = np.zeros(nrows)
        p = dataSet.Pri
        for i in range (num_days,nrows):
            pChg[i] = p[i] > p[i-num_days]
        feature_dict[column_name] = 'Keep'
        dataSet[column_name] = pChg
        return dataSet, feature_dict
    
    def LowerClose(self,dataSet,num_days,feature_dict):
        # Returns true if closing price greater than closeing price num_days previous
        column_name = str(num_days) + 'dLowerCls'
        nrows = dataSet.shape[0]
        pChg = np.zeros(nrows)
        p = dataSet.Pri
        for i in range (num_days,nrows):
            pChg[i] = p[i] < p[i-num_days]
        feature_dict[column_name] = 'Keep'
        dataSet[column_name] = pChg
        return dataSet, feature_dict      
    
if __name__ == "__main__":
    from plot_utils import *
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    candle_ind = CandleIndicators()
    plotIt = PlotUtility()
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    days_to_plot = 4
    for i in range(1,days_to_plot+1):
        num_days = i
        dataSet, feature_dict = candle_ind.HigherClose(dataSet,num_days,feature_dict)
        dataSet, feature_dict = candle_ind.LowerClose(dataSet,num_days,feature_dict)
    
    startDate = "2015-02-01"
    endDate = "2015-04-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    #fig = plt.figure(figsize=(15,8  ))
    fig, axes = plt.subplots(days_to_plot+1,1, figsize=(15,8), sharex=True)

    axes[0].plot(rsiDataSet['Pri'], label=issue)
    for i in range(1,days_to_plot+1):
        axes[i].plot(rsiDataSet[str(i) + 'dHigherCls'], label=str(i) + 'dHigherCls');

    
    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0)
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
    
    
    fig, axes = plt.subplots(days_to_plot+1,1, figsize=(15,8), sharex=True)
    axes[0].plot(rsiDataSet['Pri'], label=issue)
    for i in range(1,days_to_plot+1):
        axes[i].plot(rsiDataSet[str(i) + 'dLowerCls'], label=str(i) + 'dLowerCls');
    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0)
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
    