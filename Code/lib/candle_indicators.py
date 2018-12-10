# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
candle_indicators.py
"""
import numpy as np
from Code.lib.config import current_feature, feature_dict

class CandleIndicators:
    """Group of custom candle-based indicator features"""
    def higher_close(self, df, num_days):
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
        col_name = str(num_days) + 'dHigherCls'
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        nrows = df.shape[0]
        pChg = np.zeros(nrows)
        p = df.Close
        for i in range(num_days, nrows):
            pChg[i] = p[i] > p[i-num_days]
        df[col_name] = pChg
        return df

    def lower_close(self, df, num_days):
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
        col_name = str(num_days) + 'dLowerCls'
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        nrows = df.shape[0]
        pChg = np.zeros(nrows)
        p = df.Close
        for i in range(num_days, nrows):
            pChg[i] = p[i] < p[i-num_days]
        df[col_name] = pChg
        return df

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
        dataSet = candle_ind.higher_close(dataSet, num_days)
        dataSet = candle_ind.lower_close(dataSet, num_days)

    startDate = "2015-02-01"
    endDate = "2015-03-30"
    plotDF = dataSet[startDate:endDate]
        
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
