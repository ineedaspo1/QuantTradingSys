# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
ta_momentum_studies.py
"""
import talib as ta
from Code.lib.config import current_feature, feature_dict

class TALibMomentumStudies:
    """Group of Momentum studies utilized fromTALib """
    def RSI(self, df, period):
        """Relative Strenth Index, suppose Welles Wilder verison
           Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
           Return:
                RSI signal
                feature_dict
        """
        col_name = 'RSI_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df[col_name] = ta.RSI(df.Close, period)
        return df

    def PPO(self, df, fast, slow):
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
        col_name = 'PPO_f' + str(fast) + '_s' + str(slow)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df[col_name] = ta.PPO(df.Close,
                              # defaults are 0
                              # The FastLimit and SlowLimit parameters
                              # should be between 0.01 and 0.99
                              fast,
                              slow
                              )
        return df

    def CMO(self, df, period):
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
        col_name = 'CMO_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df[col_name] = ta.CMO(df.Close, period)
        return df

    def CCI(self, df, period):
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
        col_name = 'CCI_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df[col_name] = ta.CCI(df.High,
                              df.Low,
                              df.Close,
                              period
                              )
        return df

    def UltOsc(self, df, t1=7, t2=14, t3=28):
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
        col_name = t1t + t2t + t3t
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.ULTOSC(df.High,
                                 df.Low,
                                 df.Close,
                                 t1, t2, t3
                                 )
        return df

    def rate_OfChg(self, df, period):
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
        col_name = 'ROC_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df[col_name] = ta.ROC(df.Close, period)
        return df

if __name__ == "__main__":
    from Code.lib.plot_utils import PlotUtility
    from Code.lib.retrieve_data import DataRetrieve
    
    
    plotIt = PlotUtility()
    taLibMomSt = TALibMomentumStudies()
    dSet = DataRetrieve()
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"

    dataSet = dSet.read_issue_data(issue)

    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, dataLoadEndDate)

    dataSet = taLibMomSt.RSI(dataSet, 2)
    dataSet = taLibMomSt.PPO(dataSet, 12, 26)
    dataSet = taLibMomSt.CMO(dataSet, 20)
    dataSet = taLibMomSt.CCI(dataSet, 20)
    dataSet = taLibMomSt.UltOsc(dataSet, 7, 24, 28)
    dataSet = taLibMomSt.rate_OfChg(dataSet, 10)

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    
    # Set up dictionary and plot HigherClose
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
