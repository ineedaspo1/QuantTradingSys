# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
ta_volatility_studies.py
"""
import sys

from plot_utils import *
from retrieve_data import *
from ta_overlap_studies import *

import talib as ta
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as tkr


class TALibVolatilityStudies:
    """Group of Volatility studies utilized fromTALib """
    def ATR(self, high, low, close, period, feature_dict):
        """Average True Range (ATR) is an indicator that measures volatility.
           Args:
                high, low, close: hlc of issue
                period: timeframe of analysis
                feature_dict: Dictionary of added features
           Return:
                averageTR
                feature_dict
        """
        feature_dict['ATR_' + str(period)] = 'Keep'
        averageTR = ta.ATR(high,
                           low,
                           close,
                           period
                           )
        return averageTR, feature_dict

    def NATR(self, high, low, close, period, feature_dict):
        """Normalized Average True Range (ATR) is an indicator
        that measures volatility. NATR attempts to normalize the
        average true range values across instruments
           Args:
                high, low, close: hlc of issue
                period: timeframe of analysis
                feature_dict: Dictionary of added features
           Return:
                nATR
                feature_dict
        """
        feature_dict['NormalizedATR_' + str(period)] = 'Keep'
        nATR = ta.NATR(high,
                       low,
                       close,
                       period
                       )
        return nATR, feature_dict

    def ATR_Ratio(self, high, low, close, short, long, feature_dict):
        """ATR_Ratio is the ratio between a long-term ATR and a
        short-Term ATR
           Args:
                high, low, close: hlc of issue
                shortperiod: length of short ATR
                longperiod: length of long ATR
                feature_dict: Dictionary of added features
           Return:
                atr_Ratio
                feature_dict
        """
        feature_dict['ATRratio_S' + str(short) + "_L" + str(long)] = 'Keep'
        shortATR = ta.ATR(high,
                          low,
                          close,
                          short
                          )
        longATR = ta.ATR(high,
                         low,
                         close,
                         long
                         )
        atr_Ratio = shortATR / longATR
        return atr_Ratio, feature_dict

    def delta_ATR_Ratio(self, high, low, close,
                        short, long, delta, feature_dict):
        """Delta_ATR_Ratio is the difference between a long-term ATR and a
        short-Term ATR
           Args:
                high, low, close: hlc of issue
                shortperiod: length of short ATR
                longperiod: length of long ATR
                feature_dict: Dictionary of added features
           Return:
                delta_atr_Ratio
                feature_dict
        """
        temp_dict = {}
        deltshrt = 'DeltaATRratio_S' + str(short)
        deltlong = '_L' + str(long) + '_D' + str(delta)
        feature_dict[deltshrt + deltlong] = 'Keep'
        current_ATR_Ratio, temp_dict = self.ATR_Ratio(high,
                                                      low,
                                                      close,
                                                      short,
                                                      long,
                                                      temp_dict
                                                      )
        nrows = current_ATR_Ratio.shape[0]
        delta_atr = np.zeros(nrows)
        for i in range(delta, nrows):
            delta_atr[i] = current_ATR_Ratio[i] - current_ATR_Ratio[i-1]
        return delta_atr, feature_dict

    def BBWidth(self, close, period, feature_dict):
        """BBWidth is the width between the upper and lower BBands
           Args:
                close: close of issue
                period: length of analysis
                feature_dict: Dictionary of added features
           Return:
                bollBandWidth
                feature_dict
        """
        feature_dict['BollBandWidth_' + str(period)] = 'Keep'
        taLibOS = TALibOverlapStudies()
        uB, mB, lB, feature_dict = taLibOS.boll_bands(close,
                                                      period,
                                                      2,
                                                      feature_dict
                                                      )
        bollBandWidth = ((uB - lB) / mB) * 100
        return bollBandWidth, feature_dict

if __name__ == "__main__":
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}

    vStud = TALibVolatilityStudies()

    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)

    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, dataLoadEndDate)

    dataSet['ATR_20'], feature_dict = vStud.ATR(dataSet.High.values,
                                                dataSet.Low.values,
                                                dataSet.Pri.values,
                                                20,
                                                feature_dict
                                                )

    dataSet['NormATR_14'], feature_dict = vStud.NATR(dataSet.High.values,
                                                     dataSet.Low.values,
                                                     dataSet.Pri.values,
                                                     14,
                                                     feature_dict
                                                     )

    dataSet['ATRRatio_S10_L20'], feature_dict = vStud.ATR_Ratio(
                                                    dataSet.High.values,
                                                    dataSet.Low.values,
                                                    dataSet.Pri.values,
                                                    10,
                                                    20,
                                                    feature_dict
                                                    )

    dataSet['DeltaATRRatio_S10_L20_D5'], feature_dict = vStud.delta_ATR_Ratio(
                                                    dataSet.High.values,
                                                    dataSet.Low.values,
                                                    dataSet.Pri.values,
                                                    10,
                                                    20,
                                                    5,
                                                    feature_dict
                                                    )

    dataSet['BBWidth_20'], feature_dict = vStud.BBWidth(dataSet.Pri.values,
                                                        20,
                                                        feature_dict
                                                        )

    startDate = "2015-02-01"
    endDate = "2015-04-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    plt.figure(figsize=(18, 15))
    horizplots = 8
    top = plt.subplot2grid((horizplots, 4), (0, 0), rowspan=2, colspan=4)
    middle = plt.subplot2grid((horizplots, 4), (2, 0), rowspan=1, colspan=4)
    middle2 = plt.subplot2grid((horizplots, 4), (3, 0), rowspan=1, colspan=4)
    middle3 = plt.subplot2grid((horizplots, 4), (4, 0), rowspan=1, colspan=4)
    middle4 = plt.subplot2grid((horizplots, 4), (5, 0), rowspan=1, colspan=4)
    middle5 = plt.subplot2grid((horizplots, 4), (6, 0), rowspan=1, colspan=4)
    bottom = plt.subplot2grid((horizplots, 4), (7, 0), rowspan=1, colspan=4)

    N = len(rsiDataSet)
    ind = np.arange(N)  # the evenly spaced plot indices
    
    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return rsiDataSet.Date[thisind].strftime('%Y-%m-%d')

    top.plot(ind, rsiDataSet['Pri'], 'k-', markersize=3, label=issue)
    middle.plot(ind, rsiDataSet['ATR_20'], 'g-')
    middle2.plot(ind, rsiDataSet['NormATR_14'], '-')
    middle3.plot(ind, rsiDataSet['ATRRatio_S10_L20'], 'b-')
    middle4.plot(ind, rsiDataSet['DeltaATRRatio_S10_L20_D5'], 'r-')
    middle5.plot(ind, rsiDataSet['BBWidth_20'], 'r-')
    bottom.bar(ind, rsiDataSet['Volume'], label='Volume')

    plt.subplots_adjust(hspace=0.05)
    # set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title(str(issue))
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in middle, middle2, middle3, middle4, middle5:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.02f}'))
#        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.tick_params(axis='x', rotation=70)
#        ax.xaxis_date()
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()
