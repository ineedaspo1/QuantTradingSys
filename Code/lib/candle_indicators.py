# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
candle_indicators.py
"""
import numpy as np


class CandleIndicators:
    """Group of custom candle-based indicator features"""
    def higher_close(self, dataSet, num_days, feature_dict):
        """Returns true if closing price greater than closing
        price num_days previous
           Args:
                dataSet: dataframe of analysis
                num_days: number of days to look back
                feature_dict: Dictionary of added features
           Return:
                dataSet
                feature_dict
        """
        column_name = str(num_days) + 'dHigherCls'
        feature_dict[column_name] = 'Keep'
        nrows = dataSet.shape[0]
        pChg = np.zeros(nrows)
        p = dataSet.Pri
        for i in range(num_days, nrows):
            pChg[i] = p[i] > p[i-num_days]
        dataSet[column_name] = pChg
        return dataSet, feature_dict

    def lower_close(self, dataSet, num_days, feature_dict):
        """Returns true if closing price lesser than closing
        price num_days previous
           Args:
                dataSet: dataframe of analysis
                num_days: number of days to look back
                feature_dict: Dictionary of added features
           Return:
                dataSet
                feature_dict
        """
        column_name = str(num_days) + 'dLowerCls'
        nrows = dataSet.shape[0]
        pChg = np.zeros(nrows)
        p = dataSet.Pri
        for i in range(num_days, nrows):
            pChg[i] = p[i] < p[i-num_days]
        feature_dict[column_name] = 'Keep'
        dataSet[column_name] = pChg
        return dataSet, feature_dict

if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}

    candle_ind = CandleIndicators()
    plotIt = PlotUtility()
    dSet = DataRetrieve()
    
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )

    days_to_plot = 4
    for i in range(1, days_to_plot + 1):
        num_days = i
        dataSet, feature_dict = candle_ind.higher_close(dataSet,
                                                        num_days,
                                                        feature_dict
                                                        )
        dataSet, feature_dict = candle_ind.lower_close(dataSet,
                                                       num_days,
                                                       feature_dict
                                                       )

    startDate = "2015-02-01"
    endDate = "2015-03-30"
    plotDF = dataSet[startDate:endDate]
        
    # Set up dictionary and plot HigherClose
    plot_dict = {}
    plot_dict['Issue'] = issue
    for i in range(1, days_to_plot + 1):
        plot_dict.setdefault('Plot_Vars', []).append(str(i) + 'dHigherCls')
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
    
    # Set up dictionary and plot LowerClose
    plot_dict = {}
    plot_dict['Issue'] = issue
    for i in range(1, days_to_plot + 1):
        plot_dict.setdefault('Plot_Vars', []).append(str(i) + 'dLowerCls')
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
