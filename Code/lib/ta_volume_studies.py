# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

ta_volume_studies.py
"""

import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import *
from retrieve_data import *

import talib as ta
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class TALibVolumeStudies:
    
    def ChaikinAD(self, high, low, close, volume, feature_dict):
        feature_dict['ChaikinAD']='Keep'
        chaikinAD = ta.AD(
            high,
            low,
            close, 
            volume)
        return chaikinAD, feature_dict
    
    def ChaikinADOSC(self, high, low, close, volume, fastperiod, slowperiod, feature_dict):
        feature_dict['ChaikinADOSC_f'+str(fastperiod)+'_s'+str(slowperiod)]='Keep'
        chaikinADOSC = ta.ADOSC(
            high,
            low,
            close, 
            volume,
            fastperiod,
            slowperiod)
        return chaikinADOSC, feature_dict
    
    def OBV(self, close, volume, feature_dict):
        feature_dict['OBV']='Keep'
        onBalVol = ta.OBV(
            close, 
            volume)
        return onBalVol, feature_dict
    
    def MFI(self, dataSet, period, feature_dict):
        """Money Flow Index (MFI)
        Uses both price and volume to measure buying and selling pressure. It is
        positive when the typical price rises (buying pressure) and negative when
        the typical price declines (selling pressure). A ratio of positive and
        negative money flow is then plugged into an RSI formula to create an
        oscillator that moves between zero and one hundred.
        http://stockcharts.com/school/doku.php?
        id=chart_school:technical_indicators:money_flow_index_mfi
        Args:
            dataSet: Price series dataet
            n(int): n period.
        Returns:
            dataSet: Dataset with new feature generated.
        """
        feature_dict['MFI_'+str(period)]='Keep'
        dataSet['Up_or_Down'] = 0
        dataSet.loc[(dataSet['Close'] > dataSet['Close'].shift(1)), 'Up_or_Down'] = 1
        dataSet.loc[(dataSet['Close'] < dataSet['Close'].shift(1)), 'Up_or_Down'] = 2

        # 1 typical price
        tp = (dataSet['High'] + dataSet['Low'] + dataSet['Close']) / 3.
        
        # 2 money flow
        mf = tp * dataSet['Volume']

        # 3 positive and negative money flow with n periods
        dataSet['1p_Positive_Money_Flow'] = 0.0
        dataSet.loc[dataSet['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
        n_positive_mf = dataSet['1p_Positive_Money_Flow'].rolling(period).sum()

        dataSet['1p_Negative_Money_Flow'] = 0.0
        dataSet.loc[dataSet['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
        n_negative_mf = dataSet['1p_Negative_Money_Flow'].rolling(period).sum()
        dataSet = dataSet.drop(['1p_Positive_Money_Flow','Up_or_Down','1p_Negative_Money_Flow'], axis=1)  
        
        # 4 money flow index
        mr = n_positive_mf / n_negative_mf
        mr = (100 - (100 / (1 + mr)))
        dataSet['MFI'] = mr
        
        return dataSet, feature_dict    
    
if __name__ == "__main__":
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    taLibVolSt = TALibVolumeStudies()
        
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
        
    dataSet['ChaikinAD'], feature_dict = taLibVolSt.ChaikinAD(dataSet.High.values, dataSet.Low.values, dataSet.Pri.values, dataSet.Volume, feature_dict)
    dataSet['ChaikinADOSC'], feature_dict = taLibVolSt.ChaikinADOSC(dataSet.High.values, dataSet.Low.values, dataSet.Pri.values, dataSet.Volume, 3, 10, feature_dict)
    dataSet['OBV'], feature_dict = taLibVolSt.OBV(dataSet.Pri.values, dataSet.Volume, feature_dict)
    # MFI
    dataSet, feature_dict = taLibVolSt.MFI(dataSet, 14, feature_dict)
    
    startDate = "2014-02-01"
    endDate = "2016-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    plt.figure(figsize=(18,12))
    
    hplts = 7
    n=0
    top = plt.subplot2grid((hplts,4), (n, 0), rowspan=2, colspan=4)
    top.plot(rsiDataSet.index, rsiDataSet['Pri'], 'k-', markersize=3,label=issue)
    
    m1 = plt.subplot2grid((hplts,4), (n+2, 0), rowspan=1, colspan=4)
    m1.plot(rsiDataSet.index, rsiDataSet['ChaikinAD'], 'g-')
    m2 = plt.subplot2grid((hplts,4), (n+3, 0), rowspan=1, colspan=4)
    m2.plot(rsiDataSet.index, rsiDataSet['ChaikinADOSC'], '-')
    m3 = plt.subplot2grid((hplts,4), (n+4, 0), rowspan=1, colspan=4)
    m3.plot(rsiDataSet.index, rsiDataSet['OBV'], 'b-')
    m4= plt.subplot2grid((hplts,4), (n+5, 0), rowspan=1, colspan=4)
    m4.plot(rsiDataSet.index, rsiDataSet['MFI'], '-')
    bottom = plt.subplot2grid((hplts,4), (n+6, 0), rowspan=1, colspan=4)
    bottom.bar(rsiDataSet.index, rsiDataSet['Volume'], label='Volume')    
   
    plt.subplots_adjust(hspace=0.05)
    # set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')
    
    #top.axhline(y=30, color='red', linestyle='-', alpha=0.4)
    #top.axhline(y=70, color='blue', linestyle='-', alpha=0.4)
    m2.axhline(y=0, color='black', linestyle='-', alpha=0.4)
    
    for ax in top, m1, m2, m3, m4, bottom:
                    ax.label_outer()
                    ax.legend(loc='upper left', frameon=True, fontsize=8)
                    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax.grid(True, which='both')
                    ax.xaxis_date()
                    ax.autoscale_view()
                    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
                    ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
                    ax.minorticks_on()