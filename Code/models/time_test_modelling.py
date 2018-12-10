# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

time_test_modelling.py
Goal: Test and verify in sample and out of sample time splits for dataset
"""
from Code.lib.plot_utils import PlotUtility
from Code.lib.time_utils import TimeUtility
from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
from Code.utilities.stat_tests import stationarity_tests, mean_and_variance

import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay



if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    dSet = DataRetrieve()
    ct = ComputeTarget()
    doPlot = False
    
    issue = "tlt" 
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 2
    oos_months = 3
    segments = 3
    
    # get segmented dates
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]  
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]
    
    #load data
    dataSet = dSet.read_issue_data(issue)   
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, pivotDate)
    
    #set beLong level
    beLongThreshold = 0.0
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    
#    def stationarity_tests(s_df, signal, issue):
#        print("=============================================================")
#        adf_test(s_df, signal, issue)
#        hurst_setup(s_df[signal][:], issue)
#        print("========================================")
    
    for i in range(segments):
        modelData = dSet.set_date_range(dataSet,
                                        is_start_date,
                                        is_end_date
                                        )
        print ("\n\n\nIN SAMPLE")
        # Stationarity tests
        stationarity_tests(modelData, 'Close', issue)
        stationarity_tests(modelData, 'beLong', issue)
        #print_beLongs(modelData)
        if doPlot:
            plotIt.plot_beLongs("In Sample",
                                issue,
                                modelData,
                                is_start_date,
                                is_end_date
                                )

        
        is_start_date = is_start_date + relativedelta(months=oos_months) + BDay(1)
        is_end_date = is_start_date + relativedelta(months=is_months) - BDay(1)
        
        # OOS
        modelData = dSet.set_date_range(dataSet,
                                        oos_start_date,
                                        oos_end_date
                                        )
        print ("\n\n\nOUT OF SAMPLE")
        stationarity_tests(modelData, 'Close', issue)
        stationarity_tests(modelData, 'beLong', issue)
        #print_beLongs(modelData)
        if doPlot:
            plotIt.plot_beLongs("Out of Sample",
                                issue,
                                modelData,
                                oos_start_date,
                                oos_end_date
                                )

        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=oos_months) - BDay(1)
        
        
        
        
        