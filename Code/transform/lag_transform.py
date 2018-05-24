# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:47:44 2018

@author: kruegkj

lag_transform.py

Compute any lagged transformations of data

Inputs:
    1. data frame
    2. number of lags

Output:
    1. dataframe
"""

import sys
sys.path.append('../lib')
sys.path.append('../transform')
sys.path.append('../indicators')

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from retrieve_issue_data import *

def add_lag(df, lag_var, lags):
    #loop through ind_list
    indDataSet = df
    for i in range(0, lags):
        indDataSet[lag_var + "_lag" + str(i+1)] = indDataSet[lag_var].shift(i+1)
    return indDataSet

def main(): 
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    lag_var = 'Pri'
    lags = 5
    dataSet = add_lag(dataSet, lag_var, lags)
    return dataSet

if __name__ == "__main__":
    issue = "xrt"
    lookback = 16
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30" 
    dataSet = main()

    # Plot price and lags
    startDate = "2015-02-01"
    endDate = "2015-04-30"
    lagDataSet = dataSet.ix[startDate:endDate]
    
    print (lagDataSet.head(20))
    
    plt.style.use('seaborn-ticks')
    
    numSubPlots = 5
    # format the ticks
    fig, axes = plt.subplots(numSubPlots,1, figsize=(10,numSubPlots*2), sharex=True)
    
    axes[0].plot(lagDataSet['Pri'], label=issue)
    axes[1].plot(lagDataSet['Pri_lag1'], label='Pri_lag1');
    axes[2].plot(lagDataSet['Pri_lag2'], label='Pri_lag2');
    axes[3].plot(lagDataSet['Pri_lag3'], label='Pri_lag3');
    axes[4].plot(lagDataSet['Pri_lag4'], label='Pri_lag4');
    
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
           
    dataSet["percReturn"] = dataSet["Pri"].pct_change()*100
    lag_var = 'percReturn'
    lags = 5    
    dataSet = add_lag(dataSet, lag_var, lags)
    
    lagDataSet = dataSet.ix[startDate:endDate]
    
    numSubPlots = 5
    # format the ticks
    fig, axes = plt.subplots(numSubPlots,1, figsize=(10,numSubPlots*2), sharex=True)
    
    axes[0].plot(lagDataSet['Pri'], label=issue)
    axes[1].plot(lagDataSet['percReturn'], label='percReturn');
    axes[2].plot(lagDataSet['percReturn_lag1'], label='percReturn_lag1');
    axes[3].plot(lagDataSet['percReturn_lag2'], label='percReturn_lag2');
    axes[4].plot(lagDataSet['percReturn_lag3'], label='percReturn_lag3');
    
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
    
        