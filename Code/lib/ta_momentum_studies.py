# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
ta_momentum_studies.py
"""
import talib as ta

class TALibMomentumStudies:
    """Group of Momentum studies utilized fromTALib """
    def RSI(self, close, period, feature_dict):
        """Relative Strenth Index, suppose Welles Wilder verison
           Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
           Return:
                RSI signal
                feature_dict
        """
        feature_dict['RSI_' + str(period)] = 'Keep'
        relSI = ta.RSI(close, period)
        return relSI, feature_dict

    def PPO(self, close, fast, slow, feature_dict):
        """ Percentage Price Oscillator
                Difference between two moving averages of a security's price
            Args:
                close: Closing price of instrument
                fast: fast MA
                slow: slowMA
                feature_dict: Dictionary of added features
            Return:
                PPO signal
                feature_dict
        """
        feature_dict['PPO_f' + str(fast) + '_s' + str(slow)] = 'Keep'
        pricePercOsc = ta.PPO(close,
                              # defaults are 0
                              # The FastLimit and SlowLimit parameters
                              # should be between 0.01 and 0.99
                              fast,
                              slow
                              )
        return pricePercOsc, feature_dict

    def CMO(self, close, period, feature_dict):
        """ Chande Momentum Oscillator
                Modified RSI, measures momentum on both up and down days
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                CMO signal
                feature_dict
        """
        feature_dict['CMO_' + str(period)] = 'Keep'
        chandeMO = ta.CMO(close, period)
        return chandeMO, feature_dict

    def CCI(self, high, low, close, period, feature_dict):
        """ Commodity Channel Index
            CCI measures the current price level relative to an average price
            level over a given period of time. CCI is relatively high
            when prices are far above their average. CCI is relatively
            low when prices are far below their average. In this manner,
            CCI can be used to identify overbought and oversold levels.

            Args:
                high, low, close: HLC of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                CCI signal
                feature_dict
        """
        feature_dict['CCI_' + str(period)] = 'Keep'
        commChanIndex = ta.CCI(high,
                               low,
                               close,
                               period
                               )
        return commChanIndex, feature_dict

    def UltOsc(self, high, low, close, t1, t2, t3, feature_dict):
        """ Ultimate Oscillator
            Uses weighted sums of three oscillators, designed to capture
            momentum across three different timeframes, each of which uses
            a different time period

            Args:
                high, low, close: HLC of instrument
                t1, t2, t3: various time periods in the calculation,
                            default: 7,14,28
                feature_dict: Dictionary of added features
            Return:
                UO signal
                feature_dict
        """
        t1t = 'UltOsc_t1' + str(t1)
        t2t = '_t2' + str(t2)
        t3t = '_t3' + str(t3)
        feature_dict[t1t + t2t + t3t] = 'Keep'
        ultOsc = ta.ULTOSC(high, low, close,
                           t1, t2, t3
                           )
        return ultOsc, feature_dict

    def rate_OfChg(self, close, period, feature_dict):
        """The Rate of Change (ROC) is a technical indicator that
        measures the percentage change between the most recent price
        and the price “n” day’s ago. The indicator fluctuates around
        the zero line.
        Args:
                close: close of instrument
                feature_dict: Dictionary of added features
            Return:
                UO signal
                feature_dict
        """
        feature_dict['ROC_' + str(period)] = 'Keep'
        ROC = ta.ROC(close, period)
        return ROC, feature_dict

if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    
    plotIt = PlotUtility()
    taLibMomSt = TALibMomentumStudies()
    dSet = DataRetrieve()
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}

    dataSet = dSet.read_issue_data(issue)

    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, dataLoadEndDate)

    dataSet['RSI_20'], feature_dict = taLibMomSt.RSI(dataSet.Pri.values,
                                                     20,
                                                     feature_dict
                                                     )
    dataSet['PPO'], feature_dict = taLibMomSt.PPO(dataSet.Pri.values,
                                                  10,
                                                  24,
                                                  feature_dict
                                                  )
    dataSet['CMO_20'], feature_dict = taLibMomSt.CMO(dataSet.Pri.values,
                                                     20,
                                                     feature_dict
                                                     )
    dataSet['CCI_20'], feature_dict = taLibMomSt.CCI(dataSet.High.values,
                                                     dataSet.Low.values,
                                                     dataSet.Pri.values,
                                                     20,
                                                     feature_dict
                                                     )
    dataSet['ULTOSC'], feature_dict = taLibMomSt.UltOsc(dataSet.High.values,
                                                        dataSet.Low.values,
                                                        dataSet.Pri.values,
                                                        7,
                                                        24,
                                                        28,
                                                        feature_dict
                                                        )
    dataSet['ROC'], feature_dict = taLibMomSt.rate_OfChg(dataSet.Pri.values,
                                                         10,
                                                         feature_dict
                                                         )

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    
    # Set up dictionary and plot HigherClose
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = ['RSI_20', 'PPO', 'CMO_20', 'CCI_20', 'ULTOSC', 'ROC']
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
