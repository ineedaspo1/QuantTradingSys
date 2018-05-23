# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:02:06 2018

@author: kruegkj

predictors_main.py

Inputs
1. Issue dataframe

Actions
1. Create indicators
2. Transform indicators
3. Lag variables

Output:
1. df

"""
import sys
sys.path.append('../lib')
sys.path.append('../transform')
sys.path.append('../indicators')

from rsi_indicator import *
from roc_indicator import *
from dpo_indicator import *
from atr_indicator import *
import matplotlib.pyplot as plt
import numpy as np

def add_indicators(df, ind_list):
    #loop through ind_list
    indDataSet = df
    i = 0
    print(len(ind_list))
    for i in ind_list:
        print(i)
        sel_ind = i[0]
        if sel_ind == 'RSI':
            indDataSet['Pri_RSI'] = RSI(indDataSet.Pri, i[1])
        elif sel_ind == 'DPO':
            indDataSet['Pri_DPO'] = DPO(indDataSet.Pri, i[1])
        elif sel_ind == 'ROC':
            indDataSet['Pri_ROC'] = ROC(indDataSet.Pri, i[1])
        elif sel_ind == 'ATR':
            indDataSet['Pri_ATR'] = ATR(indDataSet.High, indDataSet.Low, indDataSet.Pri, i[1])
        else:
            continue
    return indDataSet

def main():
#    issue = "XRT"
#    dataLoadStartDate = "1998-12-22"
#    dataLoadEndDate = "2018-01-04"
    RSI_lookback = 2.3
    ROC_lookback = 5
    DPO_lookback = 5
    ATR_lookback = 5
    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    ind_list = [("RSI", RSI_lookback),("ROC",ROC_lookback),("DPO",DPO_lookback),("ATR", ATR_lookback)]
    dataSet = add_indicators(dataSet, ind_list)
    return dataSet

if __name__ == "__main__":
    dataSet = main()
    
    # Plot price and RSI indicator
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    #fig = plt.figure(figsize=(15,8  ))
    fig, axes = plt.subplots(5,1, figsize=(15,8), sharex=True)

    axes[0].plot(rsiDataSet['Pri'], label=issue)
    axes[1].plot(rsiDataSet['Pri_RSI'], label='RSI');
    axes[2].plot(rsiDataSet['Pri_ROC'], label='ROC');
    axes[3].plot(rsiDataSet['Pri_DPO'], label='DPO');
    axes[4].plot(rsiDataSet['Pri_ATR'], label='ATR');
    
    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0)
    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axes:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=False)