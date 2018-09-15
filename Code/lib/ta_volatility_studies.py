# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
ta_volatility_studies.py
"""
import talib as ta
from config import *

class TALibVolatilityStudies:
    """Group of Volatility studies utilized fromTALib """
    def ATR(self, df, period):
        """Average True Range (ATR) is an indicator that measures volatility.
           Args:
                high, low, close: hlc of issue
                period: timeframe of analysis
                feature_dict: Dictionary of added features
           Return:
                averageTR
                feature_dict
        """
        col_name = 'ATR_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        df[col_name] = ta.ATR(df.High,
                              df.Low,
                              df.Close,
                              period
                              )
        return df

    def NATR(self, df, period):
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
        col_name = 'NormATR_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        df[col_name] = ta.NATR(df.High,
                               df.Low,
                               df.Close,
                               period
                               )
        return df

    def ATR_Ratio(self, df, short, long):
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
        col_name = 'ATRratio_S' + str(short) + "_L" + str(long)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        shortATR = ta.ATR(df.High,
                          df.Low,
                          df.Close,
                          short
                          )
        longATR = ta.ATR(df.High,
                         df.Low,
                         df.Close,
                         long
                         )
        df[col_name] =  longATR / shortATR
        return df

    def delta_ATR_Ratio(self, df, short, long):
        """Delta_ATR_Ratio is the difference between a long-term ATR and a
        short-Term ATR
           Args:
                high, low, close: hlc of issue
                shortperiod: length of short ATR
                longperiod: length of long ATR
           Return:
                delta_atr_Ratio
                feature_dict
        """
        deltshrt = 'DeltaATRratio_S' + str(short)
        deltlong = '_L' + str(long)
        col_name = deltshrt + deltlong
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        shortATR = ta.ATR(df.High,
                          df.Low,
                          df.Close,
                          short
                          )
        longATR = ta.ATR(df.High,
                         df.Low,
                         df.Close,
                         long
                         )
        df[col_name] =  longATR - shortATR
        return df

    def BBWidth(self, df, period):
        """BBWidth is the width between the upper and lower BBands
           Args:
                close: close of issue
                period: length of analysis
                feature_dict: Dictionary of added features
           Return:
                bollBandWidth
                feature_dict
        """
        col_name = 'BollBandWidth_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        df['BBUB'], df['BBMB'], df['BBLB'] = ta.BBANDS(df.Close,
                                                       period
                                                       )
        df[col_name] = ((df.BBUB - df.BBLB) / df.BBMB) * 100
        return df

if __name__ == "__main__":
    import sys
    from plot_utils import *
    from retrieve_data import *
    from ta_overlap_studies import *
    from config import *
    
    vStud = TALibVolatilityStudies()
    dSet = DataRetrieve()
    plotIt = PlotUtility()
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"

    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )
    dataSet = vStud.ATR(dataSet, 20)
    dataSet = vStud.NATR(dataSet, 14,)
    dataSet = vStud.ATR_Ratio(dataSet, 10, 20)
    dataSet = vStud.delta_ATR_Ratio(dataSet, 10, 20)
    dataSet = vStud.BBWidth(dataSet, 20)

    startDate = "2015-02-01"
    endDate = "2015-04-30"
    
    plotDF = dataSet[startDate:endDate]
    
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
