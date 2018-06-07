# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:03:09 2018

@author: KRUEGKJ

transformers.py
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Transformers:
    
    def zScore_transform(self, df, zs_lb, ind):
        #loop through ind_list
        indName = str(ind+'_zScore')
        print(indName)
        df[indName] = self.zScore(df[ind], zs_lb)
        return df

    def zScore(self, p,lb):
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
        #st = pd.rolling_std(p,lb)
        st = p.rolling(window=lb,center=False).std()
        #ma = pd.rolling_mean(p,lb)
        ma = p.rolling(window=lb,center=False).mean()
        z = np.zeros(nrows)
        for i in range(lb,nrows):
            z[i] = (p[i]-ma[i])/st[i]
        return z

    def add_lag(self, df, lag_var, lags):
        #loop through ind_list
        indDataSet = df
        for i in range(0, lags):
            indDataSet[lag_var + "_lag" + str(i+1)] = indDataSet[lag_var].shift(i+1)
        return indDataSet
    
    def softmax(self, p,lb,lam):
        # softmax transformation.
        # p, the series being transformed.
        # lb, the lookback period, an integer.
        #     the length used for the average and standard deviation.
        #     typical values 20 to 252.  Be aware of ramp-up requirement.
        # lam, the length of the linear section.
        #     in standard deviations.
        #     typical value is 6.
        # Return is a numpy array with values in the range 0.0 to 1.0.
        nrows = p.shape[0]
        a = np.zeros(nrows)
        ma = np.zeros(nrows)
        sd = np.zeros(nrows)    
        sm = np.zeros(nrows)
        sq = np.zeros(nrows)
        y = np.zeros(nrows)
        for i in range(lb,nrows):
            sm[i] = sm[i]+p[i]
        ma[i] = sm[i] / lb
        for i in range(lb,nrows):
            sq[i] = (p[i]-ma[i])*(p[i]-ma[i])
        sd[i] = math.sqrt(sq[i]/(nrows-1))
        for i in range(lb,nrows):
            a[i] = (p[i]-ma[i])/((lam*sd[i])/(2.0*math.pi))
            y[i] = 1.0 / (1.0 + math.e**a[i])
        return y
    
if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    from indicators import *
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    addIndic1 = Indicators()
    ind_list = [("RSI", 2.3),("ROC",5),("DPO",5),("ATR", 5)]
    dataSet = addIndic1.add_indicators(dataSet, ind_list)
 
    zScore_lookback = 3
    transf = Transformers()
    transfList = ['Pri_ROC','Pri_DPO','Pri_ATR']
    for i in transfList:
        dataSet = transf.zScore_transform(dataSet, zScore_lookback, i)

    # Plot price and indicators
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    numSubPlots = 8
    # format the ticks
    fig, axes = plt.subplots(numSubPlots,1, figsize=(10,numSubPlots*2), sharex=True)
    #fig, axes = plt.subplots(8,1, figsize=(15,8), sharex=True)
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
            ax.legend(loc='upper left', frameon=True, fontsize=8)
            ax.grid(True, which='both')
            fig.autofmt_xdate()
            ax.xaxis_date()
            ax.autoscale_view()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            ax.minorticks_on()
            ax.tick_params(axis='y',which='minor',bottom='off')
    
    #### testing Lag
    lag_var = 'Pri'
    lags = 5
    dataSet = transf.add_lag(dataSet, lag_var, lags)
    
    # Plot price and lags
    startDate = "2015-02-01"
    endDate = "2015-04-30"
    lagDataSet = dataSet.ix[startDate:endDate]
    
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