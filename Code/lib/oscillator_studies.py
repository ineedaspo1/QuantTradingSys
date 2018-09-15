# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
oscillator_studies.py
"""
import numpy as np
from config import *


class OscialltorStudies:
    """Group of oscillator-based indicator features"""
    def detrend_PO(self, df, col, lb):
        """<add content here>
           Args:
                p: data series
                lb: number of days to look back
                feature_dict: Dictionary of added features
           Return:
                d: numpy array with values centered on 0.0.
                feature_dict
        """
        col_name = 'DPO_' + str(lb)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        p = df[col]
        nrows = p.shape[0]
        ma = p.ewm(span=lb,
                   min_periods=0,
                   adjust=True,
                   ignore_na=False).mean()
        d = np.zeros(nrows)
        for i in range(1, nrows):
            d[i] = (p[i] - ma[i]) / ma[i]
        df[col_name] = d
        return df

if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    
    dSet = DataRetrieve()
    oscSt = OscialltorStudies()
    plotIt = PlotUtility()
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )

    dataSet = oscSt.detrend_PO(dataSet, 'Close', 10)
    dataSet = oscSt.detrend_PO(dataSet, 'Close', 50)

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)