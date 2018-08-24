# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

ta_overlap_studies.py
"""

import sys
#sys.path.append('../lib')
#sys.path.append('../utilities')

from plot_utils import *
from retrieve_data import *

import talib as ta
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class TALibOverlapStudies:
    
    def BollBands(self, close, period, nbdev=2, feature_dict={}):
        stddev = nbdev
        bollUPPER, bollMIDDLE, bollLOWER = ta.BBANDS(
            # Closing price, narray
            close,
            # time def 20
            timeperiod = period,
            # num of std deviations from mean
            nbdevup = stddev,
            nbdevdn = stddev,
            # Moving average type: 0-simple
            matype=0)
        feature_dict['BollBands_'+str(period)]='Keep'
        return bollUPPER, bollMIDDLE, bollLOWER, feature_dict
    
    def ExpMA(self, close, period, feature_dict):
        feature_dict['EMA_'+str(period)]='Keep'
        expMA = ta.EMA(
            close,
            # default is 30
            period)
        return expMA, feature_dict
    
    def SimpleMA(self, close, period, feature_dict):
        feature_dict['SMA_'+str(period)]='Keep'
        simpleMA = ta.SMA(
            close,
            # default is 30
            period)
        return simpleMA, feature_dict
    
    def WeightedMA(self, close, period, feature_dict):
        feature_dict['WgnthdMA_'+str(period)]='Keep'
        weightedMA = ta.WMA(
            close,
            # default is 30
            period)
        return weightedMA, feature_dict
    
    def TripleEMA(self, close, period, feature_dict):
        feature_dict['TripleMA_'+str(period)]='Keep'
        tripleEMA = ta.TEMA(
            close,
            # default is 30
            period)
        return tripleEMA, feature_dict
    
    def TriangularMA(self, close, period, feature_dict):
        feature_dict['TriangMA_'+str(period)]='Keep'
        triangularMA = ta.TRIMA(
            close,
            # default is 30
            period)
        return triangularMA, feature_dict
    
    def DblExpMA(self, close, period, feature_dict):
        feature_dict['DblExpMA_'+str(period)]='Keep'
        dblExpMA = ta.DEMA(
            close,
            # default is 30
            period)
        return dblExpMA, feature_dict

    def KaufmanAMA(self, close, period, feature_dict):
        feature_dict['KAMA_'+str(period)]='Keep'
        kaufmanAMA = ta.KAMA(
            close,
            # default is 30
            period)
        return kaufmanAMA, feature_dict
    
    def MesaAMA(self, close, fastlimit, slowlimit, feature_dict):
        feature_dict['MesaAMA_f'+str(fastlimit)+'_s'+str(slowlimit)]='Keep'
        mama, fama = ta.MAMA(
            close,
            # defaults are 0
            # The FastLimit and SlowLimit parameters should be between 0.01 and 0.99
            fastlimit,
            slowlimit)
        return mama, fama, feature_dict
    
    def HilbertTransform(self, close, feature_dict):
        feature_dict['HTTrendline']='Keep'
        HTTrendline = ta.HT_TRENDLINE(close)
        return HTTrendline, feature_dict
    
    def MidPoint(self, close, period, feature_dict):
        feature_dict['Midpoint_'+str(period)]='Keep'
        midPoint = ta.MIDPOINT(
            close,
            # default is 30
            period)
        return midPoint, feature_dict
    
    def MidPrice(self, high, low, period, feature_dict):
        feature_dict['Midprice_'+str(period)]='Keep'
        midPrice = ta.MIDPRICE(
            high,
            low,
            # default is 30
            period)
        return midPrice, feature_dict

    def PSAR(self, high, low, period, feature_dict):
        feature_dict['PSAR_'+str(period)]='Keep'
        pSAR = ta.SAR(
            high,
            low,
            # default is 30
            period)
        return pSAR, feature_dict
    
if __name__ == "__main__":
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    taLibOS = TALibOverlapStudies()
        
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
        
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    dataSet['BB-UB'],dataSet['BB-MB'],dataSet['BB-LB'], feature_dict = taLibOS.BollBands(dataSet.Pri.values, 20, 2, feature_dict)
    dataSet['DblExpMA_20'], feature_dict = taLibOS.DblExpMA(dataSet.Pri.values, 20, feature_dict)
    dataSet['KAMA_20'], feature_dict = taLibOS.KaufmanAMA(dataSet.Pri.values, 20, feature_dict)
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    plt.figure(figsize=(15,8))
    top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
    #top.plot(rsiDataSet.index, rsiDataSet['Pri']) #
    #bottom.bar(rsiDataSet.index, rsiDataSet['Volume'])
    top.plot(rsiDataSet.index, rsiDataSet['Pri'], 'k-', markersize=3,label=issue)
    top.plot(rsiDataSet.index, rsiDataSet['BB-UB'], 'c-')
    top.plot(rsiDataSet.index, rsiDataSet['BB-MB'], 'c--')
    top.plot(rsiDataSet.index, rsiDataSet['BB-LB'], 'c-')
    top.plot(rsiDataSet.index, rsiDataSet['DblExpMA_20'], 'y-')
    top.plot(rsiDataSet.index, rsiDataSet['KAMA_20'], 'g-')
    bottom.bar(rsiDataSet.index, rsiDataSet['Volume'], label='Volume');
    plt.subplots_adjust(hspace=0.05)
    # set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')
    
    for ax in top, bottom:
                    ax.label_outer()
                    ax.legend(loc='upper left', frameon=True, fontsize=8)
                    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax.grid(True, which='both')
                    ax.xaxis_date()
                    ax.autoscale_view()
                    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
                    ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
                    ax.minorticks_on()
                    
    dataSet['MesaAMA_Fast'], dataSet['MesaAMA_Slow'], feature_dict = taLibOS.MesaAMA(dataSet.Pri.values, 0.9, 0.1, feature_dict)
    dataSet['EMA_30'], feature_dict = taLibOS.ExpMA(dataSet.Pri.values, 30, feature_dict)
    dataSet['HT_Trendline'], feature_dict = taLibOS.HilbertTransform(dataSet.Pri.values, feature_dict)
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    plt.figure(figsize=(15,8))
    top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
    #top.plot(rsiDataSet.index, rsiDataSet['Pri']) #
    #bottom.bar(rsiDataSet.index, rsiDataSet['Volume'])
    top.plot(rsiDataSet.index, rsiDataSet['Pri'], 'k-', markersize=3,label=issue)
    top.plot(rsiDataSet.index, rsiDataSet['MesaAMA_Fast'], 'g-')
    top.plot(rsiDataSet.index, rsiDataSet['MesaAMA_Slow'], 'r-')
    top.plot(rsiDataSet.index, rsiDataSet['EMA_30'], 'b--', alpha=0.6)
    top.plot(rsiDataSet.index, rsiDataSet['HT_Trendline'], 'y-')
    bottom.bar(rsiDataSet.index, rsiDataSet['Volume'], label='Volume');
    plt.subplots_adjust(hspace=0.05)
    # set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')
    
    for ax in top, bottom:
                    ax.label_outer()
                    ax.legend(loc='upper left', frameon=True, fontsize=8)
                    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax.grid(True, which='both')
                    ax.xaxis_date()
                    ax.autoscale_view()
                    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
                    ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
                    ax.minorticks_on()
                    
    dataSet['Midpoint_30'], feature_dict = taLibOS.MidPoint(dataSet.Pri.values, 30, feature_dict)
    dataSet['Midprice_30'], feature_dict = taLibOS.MidPrice(dataSet.High.values, dataSet.Low.values, 30, feature_dict)
    dataSet['PSAR'], feature_dict = taLibOS.PSAR(dataSet.High.values, dataSet.Low.values, 30, feature_dict)
    dataSet['SMA_30'], feature_dict = taLibOS.SimpleMA(dataSet.Pri.values, 30, feature_dict)
    dataSet['TEMA_30'], feature_dict = taLibOS.TripleEMA(dataSet.Pri.values, 30, feature_dict)
    dataSet['TRIMA_30'], feature_dict = taLibOS.TriangularMA(dataSet.Pri.values, 30, feature_dict)
    dataSet['WMA_30'], feature_dict = taLibOS.WeightedMA(dataSet.Pri.values, 30, feature_dict)
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    plt.figure(figsize=(15,8))
    top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
    #top.plot(rsiDataSet.index, rsiDataSet['Pri']) #
    #bottom.bar(rsiDataSet.index, rsiDataSet['Volume'])
    top.plot(rsiDataSet.index, rsiDataSet['Pri'], 'k-', markersize=3,label=issue)
    top.plot(rsiDataSet.index, rsiDataSet['Midpoint_30'], 'g-')
    top.plot(rsiDataSet.index, rsiDataSet['Midprice_30'], 'y-')
    top.plot(rsiDataSet.index, rsiDataSet['PSAR'], 'r+')
    top.plot(rsiDataSet.index, rsiDataSet['SMA_30'], 'b-')
    top.plot(rsiDataSet.index, rsiDataSet['TEMA_30'], 'm-')
    top.plot(rsiDataSet.index, rsiDataSet['TRIMA_30'], 'c-')
    top.plot(rsiDataSet.index, rsiDataSet['WMA_30'], 'k:')
    bottom.bar(rsiDataSet.index, rsiDataSet['Volume'], label='Volume');
    plt.subplots_adjust(hspace=0.05)
    # set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')
    
    for ax in top, bottom:
                    ax.label_outer()
                    ax.legend(loc='upper left', frameon=True, fontsize=8)
                    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax.grid(True, which='both')
                    ax.xaxis_date()
                    ax.autoscale_view()
                    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
                    ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
                    ax.minorticks_on()