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
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm

class Transformers:
    
    def zScore_transform(self, df, zs_lb, ind, feature_dict):
        #loop through ind_list
        indName = str(ind)+'_zScore_'+str(zs_lb)
        print(indName)
        df[indName] = self.zScore(df[ind], zs_lb)
        feature_dict[indName] = 'Keep'
        return df, feature_dict

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

    def add_lag(self, df, lag_var, lags, feature_dict):
        #loop through ind_list
        indDataSet = df
        for i in range(0, lags):
            indDataSet[lag_var + "_lag" + str(i+1)] = indDataSet[lag_var].shift(i+1)
            feature_dict[lag_var + "_lag" + str(i+1)] = 'Keep'
        return indDataSet, feature_dict
    
    def Center(self, df, col, feature_dict, lb=14, type='median'):
        """
            Center - subtract a historical median from signal,
            use the pandas sliding window functions.
            Args:
                dataSet: Time series dataset
                col: Column name to be centered
                feature_dict: Dictionary of added features
                lb(int): lookback period
                type: default is median, col
            Returns:
                dataSet: Dataset with new feature generated.
                feature_dict: Append entry with colname
        """
        col_name = str(col) + '_Centered'
        feature_dict[col_name] = 'Keep'
        df[col_name] = df[col]
        if type == 'median':
            rm = df[col].rolling(window=lb, center=False).median()
            df[col_name] = df[col] - rm
        return df, feature_dict
    
    def Scale(self, df, col, feature_dict):
        """
            Scaled - Use when sign and magnitude is of paramount importance.
            Scale accoring to historical volatility defined by interquartile range.
            Args:
                df: Signal to be centered
                col: Column name to be centered
                feature_dict: Dictionary of added features
                lb(int): lookback period
                type: default is median, col
            Returns:
                dataSet: Dataset with new feature generated.
                feature_dict: Append entry with colname
        """
        col_name = str(col) + '_Scaled'
        feature_dict[col_name] = 'Keep'
        df[col_name] = df[col]
        scaler = RobustScaler(quantile_range=(25, 75))
        df[[col_name]] = scaler.fit_transform(df[[col_name]])
        return df, feature_dict



    
    def Normalize(self, dataSet, colname, n, feature_dict, mode = 'scale', linear = False):
        """
             It computes the normalized value on the stats of n values 
             ( Modes: total or scale ) using the formulas from the book 
             "Statistically sound machine learning..." (Aronson and Masters) 
             but the decission to apply a non linear scaling is left to the 
             user. It is modified to fit the data from -1 to 1 instead of 
             -100 to 100 df is an imput DataFrame. it returns also a 
             DataFrame, but it could return a list.
             n define the number of data points to get the mean and the 
             quartiles for the normalization
             modes: scale: scale, without centering. total: center and scale.
        """
        temp =[]
        new_colname = str(colname) + '_Normalized'
        feature_dict[new_colname] = 'Keep'
        
        df = dataSet[colname]
        #print(df)

        for i in range(len(df))[::-1]:

            if i  >= n: # there will be a traveling norm until we reach the initian n values. 
                        # those values will be normalized using the last computed values of F50,F75 and F25
                F50 = df[i-n:i].quantile(0.5)
                F75 =  df[i-n:i].quantile(0.75)
                F25 =  df[i-n:i].quantile(0.25)

            if linear == True and mode == 'total':
                 v = 50 * ((df.iloc[i]-F50)/(F75-F25))-50
            elif linear == True and mode == 'scale':
                 v =  25 * df.iloc[i]/(F75-F25) -50
            elif linear == False and mode == 'scale':
                 v = 100 * norm.cdf(0.5*df.iloc[i]/(F75-F25))-50

            else: # even if strange values are given, it will perform full normalization with compression as default
                v = norm.cdf(50*(df.iloc[i]-F50)/(F75-F25))-50
            #print(v)
            temp.append(v)
        #print(temp)
        dataSet[new_colname] = temp[::-1]
        return  dataSet, feature_dict
    

    
if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    from indicators import *
    from ta_volume_studies import *
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    taLibVolSt = TALibVolumeStudies()
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    addIndic1 = Indicators()
    ind_list = [("RSI", 2.3),("ROC",5),("DPO",5),("ATR", 5)]
    dataSet, feature_dict = addIndic1.add_indicators(dataSet, ind_list, feature_dict)
 
    zScore_lb = 3
    transf = Transformers()
    transfList = ['ROC','DPO','ATR']
    for i in transfList:
        dataSet, feature_dict = transf.zScore_transform(dataSet, zScore_lb, i, feature_dict)

    # Plot price and indicators
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    numSubPlots = 8
    # format the ticks
    fig, axes = plt.subplots(numSubPlots,1, figsize=(10,numSubPlots*2), sharex=True)
    #fig, axes = plt.subplots(8,1, figsize=(15,8), sharex=True)
    axes[0].plot(rsiDataSet['Pri'], label=issue)
    axes[1].plot(rsiDataSet['RSI'], label='RSI');
    axes[2].plot(rsiDataSet['ROC'], label='ROC');
    axes[3].plot(rsiDataSet['ROC_zScore_'+str(zScore_lb)], label='ROC_zScore');
    axes[4].plot(rsiDataSet['DPO'], label='DPO');
    axes[5].plot(rsiDataSet['DPO_zScore_'+str(zScore_lb)], label='DPO_zScore');
    axes[6].plot(rsiDataSet['ATR'], label='ATR');
    axes[7].plot(rsiDataSet['ATR_zScore_'+str(zScore_lb)], label='ATR_zScore');
    
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
    dataSet, feature_dict = transf.add_lag(dataSet, lag_var, lags, feature_dict)
    
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
            
    dataSet['ChaikinAD'], feature_dict = taLibVolSt.ChaikinAD(
            dataSet.High.values, 
            dataSet.Low.values, 
            dataSet.Pri.values, 
            dataSet.Volume, 
            feature_dict)
    
    dataSet, feature_dict = transf.Scale(
            dataSet, 
            'ChaikinAD', 
            feature_dict)

    dataSet, feature_dict = transf.Center(
            dataSet, 
            'ChaikinAD', 
            feature_dict, 
            14)

    dataSet, feature_dict = transf.Normalize(
            dataSet, 
            'ChaikinAD', 
            200, 
            feature_dict, 
            mode='scale', 
            linear=False)

    startDate = "2015-10-01"
    endDate = "2016-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    plt.figure(figsize=(15,10))
    horizplots = 7
    top = plt.subplot2grid((horizplots,4), (0, 0), rowspan=2, colspan=4)
    middle = plt.subplot2grid((horizplots,4), (2, 0), rowspan=1, colspan=4)
    middle2 = plt.subplot2grid((horizplots,4), (3, 0), rowspan=1, colspan=4)
    middle3 = plt.subplot2grid((horizplots,4), (4, 0), rowspan=1, colspan=4)
    middle4 = plt.subplot2grid((horizplots,4), (5, 0), rowspan=1, colspan=4)
    bottom = plt.subplot2grid((horizplots,4), (6, 0), rowspan=1, colspan=4)
    
    top.plot(rsiDataSet.index, rsiDataSet['Pri'], 'k-', markersize=3,label=issue)
    middle.plot(rsiDataSet.index, rsiDataSet['ChaikinAD'], 'g-')
    middle2.plot(rsiDataSet.index, rsiDataSet['ChaikinAD_Scaled'], '-')
    middle3.plot(rsiDataSet.index, rsiDataSet['ChaikinAD_Centered'], 'b-')
    middle4.plot(rsiDataSet.index, rsiDataSet['ChaikinAD_Normalized'], 'b-')
    bottom.bar(rsiDataSet.index, rsiDataSet['Volume'], label='Volume')
    
    plt.subplots_adjust(hspace=0.05)
    # set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')
    
    #top.axhline(y=30, color='red', linestyle='-', alpha=0.4)
    #top.axhline(y=70, color='blue', linestyle='-', alpha=0.4)
    middle.axhline(y=0, color='black', linestyle='-', alpha=0.4)
    
    for ax in top, middle, middle2, middle3, middle4, bottom:
                    ax.label_outer()
                    ax.legend(loc='upper left', frameon=True, fontsize=12)
                    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
                    ax.grid(True, which='both')
                    ax.xaxis_date()
                    ax.autoscale_view()
                    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
                    ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
                    ax.minorticks_on()
    