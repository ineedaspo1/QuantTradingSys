# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:02:06 2018

@author: kruegkj
transformers_main.py

Inputs
1. Issue dataframe with predictors

Actions
1. Idenitfy predictors to be transformed
2. Transform indicators
3. Keep list of predictors to be dropped later

Output:
1. df
2. List of transformed indicators
3. List of transformer(s) to be used

"""
import sys
sys.path.append('../lib')
sys.path.append('../indicators')
sys.path.append('../predictors')

from retrieve_issue_data import *
from softmax_transform import *
from zScore_transform import *
import predictors_main
import matplotlib.pyplot as plt
import numpy as np

def zScore_transform(df, zs_lb, ind):
    #loop through ind_list
    indName = str(ind+'_zScore')
    print(indName)
    df[indName] = zScore(df[ind], zs_lb)
    return df

def main():
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
    zScore_lookback = 3
    dataSet = zScore_transform(dataSet, zScore_lookback, 'Pri_ROC')
    dataSet = zScore_transform(dataSet, zScore_lookback, 'Pri_DPO')
    dataSet = zScore_transform(dataSet, zScore_lookback, 'Pri_ATR')
    # Plot price and indicators
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    fig, axes = plt.subplots(8,1, figsize=(15,8), sharex=True)
    axes[0].plot(rsiDataSet['Pri'], label=issue)
    axes[1].plot(rsiDataSet['Pri_RSI'], label='RSI');
    axes[2].plot(rsiDataSet['Pri_ROC'], label='ROC');
    axes[3].plot(rsiDataSet['Pri_ROC_zScore'], label='ROC_zScore');
    axes[4].plot(rsiDataSet['Pri_DPO'], label='DPO');
    axes[5].plot(rsiDataSet['Pri_DPO_zScore'], label='DPO_zScore');
    axes[6].plot(rsiDataSet['Pri_ATR'], label='ATR');
    axes[7].plot(rsiDataSet['Pri_ATR_zScore'], label='ATR_zScore');
    
    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0)
    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axes:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=False)
        ax.grid(True)
    

