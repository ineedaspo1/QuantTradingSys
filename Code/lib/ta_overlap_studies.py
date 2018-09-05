# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
ta_overlap_studies.py
"""
import sys

import talib as ta
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class TALibOverlapStudies:
    """Group of Momentum studies utilized fromTALib """

    def boll_bands(self, close, period, nbdev=2, feature_dict={}):
        """Bollinger Bands
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                nbdev = Number of std dev's
                feature_dict: Dictionary of added features
            Return:
                Upper, middle, lower prices of Bollinger Band
                feature_dict
        """
        stddev = nbdev
        bollUPPER, bollMIDDLE, bollLOWER = ta.BBANDS(
                close,
                timeperiod=period,
                nbdevup=stddev,
                nbdevdn=stddev,
                matype=0  # Moving average type: 0-simple
                )
        feature_dict['BollBands' + str(period)] = 'Keep'
        return bollUPPER, bollMIDDLE, bollLOWER, feature_dict

    def exp_MA(self, close, period, feature_dict):
        """Exponential MA
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                expMA
                feature_dict
        """
        feature_dict['EMA_' + str(period)] = 'Keep'
        expMA = ta.EMA(close,
                       period
                       )
        return expMA, feature_dict

    def simple_MA(self, close, period, feature_dict):
        """Simple MA
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                simpleMA
                feature_dict
        """
        feature_dict['SMA_' + str(period)] = 'Keep'
        simpleMA = ta.SMA(close,
                          period
                          )
        return simpleMA, feature_dict

    def weighted_MA(self, close, period, feature_dict):
        """Weighted MA calculates a weight for each value in the series.
        The more recent values are assigned greater weights.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                weightedMA
                feature_dict
        """
        feature_dict['WgnthdMA_' + str(period)] = 'Keep'
        weightedMA = ta.WMA(close,
                            period  # default is 30
                            )
        return weightedMA, feature_dict

    def triple_MA(self, close, period, feature_dict):
        """Triple MA  a smoothing indicator with less lag than a straight
        exponential moving average.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                tripleEMA
                feature_dict
        """
        feature_dict['TripleMA_' + str(period)] = 'Keep'
        tripleEMA = ta.TEMA(close,
                            period  # default is 30
                            )
        return tripleEMA, feature_dict

    def tri_ma(self, close, period, feature_dict):
        """The Triangular Moving Average is a form of Weighted Moving
        Average wherein the weights are assigned in a triangular pattern.
        For example, the weights for a 7 period Triangular Moving Average
        would be 1, 2, 3, 4, 3, 2, 1. This gives more weight to the middle
        of the time series and less weight to the oldest and newest data.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                triangularMA
                feature_dict
        """
        feature_dict['TriangMA_' + str(period)] = 'Keep'
        triangularMA = ta.TRIMA(close,
                                period  # default is 30
                                )
        return triangularMA, feature_dict

    def dbl_exp_MA(self, close, period, feature_dict):
        """The DEMA is a smoothing indicator with less lag than a straight
        exponential moving average
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                dblExpMA
                feature_dict
        """
        feature_dict['DblExpMA_' + str(period)] = 'Keep'
        dblExpMA = ta.DEMA(close,
                           period  # default is 30
                           )
        return dblExpMA, feature_dict

    def kaufman_AMA(self, close, period, feature_dict):
        """The KAMA is a moving average designed to account for market noise
        or volatility. KAMA will closely follow prices when the price swings
        are relatively small and the noise is low. KAMA will adjust when the
        price swings widen and follow prices from a greater distance. This
        trend-following indicator can be used to identify the overall trend,
        time turning points and filter price movements.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                kaufmanAMA
                feature_dict
            To Update: KAMA has 3 parameteres, but TA-Lib only exposes 1. Move
                to custom code to expose all 3
        """
        feature_dict['KAMA_' + str(period)] = 'Keep'
        kaufmanAMA = ta.KAMA(close,
                             period
                             )
        return kaufmanAMA, feature_dict

    def mesa_AMA(self, close, flimit, slimit, feature_dict):
        """The MESA Adaptive Moving Average is a technical trend-following
        indicator which, according to its creator, adapts to price movement
        “based on the rate change of phase as measured by the Hilbert Transform
        Discriminator”.
        http://www.binarytribune.com/forex-trading-indicators/ehlers-mesa-adaptive-moving-average
            Args:
                close: Closing price of instrument
                flimit: number of fast time periods in the calculation
                slimit: number of slow time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                mama: MESA Adaptive MA
                fama: MAMA beign applied to mama
                feature_dict
        """
        feature_dict['MesaAMA_f' + str(flimit) + '_s' + str(slimit)] = 'Keep'
        mama, fama = ta.MAMA(close,
                             #  defaults are 0
                             #  The FastLimit and SlowLimit parameters
                             #  should be between 0.01 and 0.99
                             flimit,
                             slimit
                             )
        return mama, fama, feature_dict

    def inst_Trendline(self, close, feature_dict):
        """The Ehlers Hilbert Transform - Instantaneous Trendline is a
         smoothed trendline,
            Args:
                close: Closing price of instrument
                feature_dict: Dictionary of added features
            Return:
                instTrendline: resulting signal
                feature_dict
        """
        feature_dict['InstTrendline'] = 'Keep'
        instTrendline = ta.HT_TRENDLINE(close)
        return instTrendline, feature_dict

    def mid_point(self, close, period, feature_dict):
        """(highest value + lowest value)/2 over period
            Args:
                close: closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                midPrice: resulting signal
                feature_dict
        """
        feature_dict['Midpoint_' + str(period)] = 'Keep'
        midPoint = ta.MIDPOINT(close,
                               period  # default is 30
                               )
        return midPoint, feature_dict

    def mid_price(self, high, low, period, feature_dict):
        """(highest high + lowest low)/2 over period
            Args:
                high, low: high, low price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                midPrice: resulting signal
                feature_dict
        """
        feature_dict['Midprice_' + str(period)] = 'Keep'
        midPrice = ta.MIDPRICE(high,
                               low,
                               period  # default is 30
                               )
        return midPrice, feature_dict

    def pSAR(self, high, low, period, feature_dict):
        """The parabolic SAR is a technical indicator used to determine the
        price direction of an asset, as well draw attention to when the price
        direction is changing.
            Args:
                high, low: high, low price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                pSAR: resulting signal
                feature_dict
        """
        feature_dict['PSAR_' + str(period)] = 'Keep'
        pSAR = ta.SAR(high,
                      low,
                      period  # default is 30
                      )
        return pSAR, feature_dict


if __name__ == "__main__":   
    from plot_utils import *
    from retrieve_data import *
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}

    taLibOS = TALibOverlapStudies()

    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)

    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )
    ub, mb, lb, feature_dict = taLibOS.boll_bands(dataSet.Pri.values,
                                                 20,
                                                 2,
                                                 feature_dict
                                                 )
    dataSet['BB-UB'] = ub
    dataSet['BB-MB'] = mb
    dataSet['BB-LB'] = lb

    dblexpma, feature_dict = taLibOS.dbl_exp_MA(dataSet.Pri.values,
                                                20,
                                                feature_dict
                                                )
    dataSet['DblExpMA_20'] = dblexpma
    dataSet['KAMA_20'], feature_dict = taLibOS.kaufman_AMA(dataSet.Pri.values,
                                                          20,
                                                          feature_dict
                                                          )

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    
    N = len(plotDF)
    ind = np.arange(N)  # the evenly spaced plot indices

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return plotDF.Date[thisind].strftime('%Y-%m-%d')
    
    plt.figure(figsize=(15, 8))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    #  top.plot(ind, plotDF['Pri']) #
    #  bottom.bar(ind, plotDF['Volume'])
    top.plot(ind,
             plotDF['Pri'],
             'k-', markersize=3,
             label=issue)
    top.plot(ind,
             plotDF['BB-UB'], 'c-')
    top.plot(ind, plotDF['BB-MB'], 'c--')
    top.plot(ind, plotDF['BB-LB'], 'c-')
    top.plot(ind, plotDF['DblExpMA_20'], 'y-')
    top.plot(ind, plotDF['KAMA_20'], 'g-')
    bottom.bar(ind, plotDF['Volume'], label='Volume')
    plt.subplots_adjust(hspace=0.05)
    #  set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()

    mfast, mslow, feature_dict = taLibOS.mesa_AMA(dataSet.Pri.values,
                                                 0.9,
                                                 0.1,
                                                 feature_dict
                                                 )
    dataSet['MesaAMA_Fast'] = mfast
    dataSet['MesaAMA_Slow'] = mslow

    dataSet['EMA_30'], feature_dict = taLibOS.exp_MA(dataSet.Pri.values,
                                                    30,
                                                    feature_dict
                                                    )
    inst_trend, feature_dict = taLibOS.inst_Trendline(dataSet.Pri.values,
                                                     feature_dict
                                                     )
    dataSet['HT_Trendline'] = inst_trend

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    N = len(plotDF)
    ind = np.arange(N)  # the evenly spaced plot indices
    plt.figure(figsize=(15, 8))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    #  top.plot(ind, plotDF['Pri']) #
    #  bottom.bar(ind, plotDF['Volume'])
    top.plot(ind,
             plotDF['Pri'],
             'k-',
             markersize=3,
             label=issue)
    top.plot(ind, plotDF['MesaAMA_Fast'], 'g-')
    top.plot(ind, plotDF['MesaAMA_Slow'], 'r-')
    top.plot(ind, plotDF['EMA_30'], 'b--', alpha=0.6)
    top.plot(ind, plotDF['HT_Trendline'], 'y-')
    bottom.bar(ind, plotDF['Volume'], label='Volume')
    plt.subplots_adjust(hspace=0.05)
    #  set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()

    dataSet['Midpt_30'], feature_dict = taLibOS.mid_point(dataSet.Pri.values,
                                                         30,
                                                         feature_dict
                                                         )
    dataSet['Midprc_30'], feature_dict = taLibOS.mid_price(dataSet.High.values,
                                                          dataSet.Low.values,
                                                          30,
                                                          feature_dict
                                                          )
    dataSet['PSAR'], feature_dict = taLibOS.pSAR(dataSet.High.values,
                                                 dataSet.Low.values,
                                                 30,
                                                 feature_dict
                                                 )
    dataSet['SMA_30'], feature_dict = taLibOS.simple_MA(dataSet.Pri.values,
                                                       30,
                                                       feature_dict
                                                       )
    dataSet['TEMA_30'], feature_dict = taLibOS.triple_MA(dataSet.Pri.values,
                                                         30,
                                                         feature_dict
                                                         )
    dataSet['TRIMA_30'], feature_dict = taLibOS.tri_ma(dataSet.Pri.values,
                                                       30,
                                                       feature_dict
                                                       )
    dataSet['WMA_30'], feature_dict = taLibOS.weighted_MA(dataSet.Pri.values,
                                                         30,
                                                         feature_dict
                                                         )

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    N = len(plotDF)
    ind = np.arange(N)  # the evenly spaced plot indices
    plt.figure(figsize=(15, 8))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    #  top.plot(ind, plotDF['Pri']) #
    #  bottom.bar(ind, plotDF['Volume'])
    top.plot(ind,
             plotDF['Pri'],
             'k-',
             markersize=3,
             label=issue
             )
    top.plot(ind, plotDF['Midpt_30'], 'g-')
    top.plot(ind, plotDF['Midprc_30'], 'y-')
    top.plot(ind, plotDF['PSAR'], 'r+')
    top.plot(ind, plotDF['SMA_30'], 'b-')
    top.plot(ind, plotDF['TEMA_30'], 'm-')
    top.plot(ind, plotDF['TRIMA_30'], 'c-')
    top.plot(ind, plotDF['WMA_30'], 'k:')
    bottom.bar(ind, plotDF['Volume'], label='Volume')
    plt.subplots_adjust(hspace=0.05)
    #  set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()
