# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

time_test_modelling.py
Goal: Test and verify in sample and out of sample time splits for dataset
"""
import sys
sys.path.append('../lib')
sys.path.append('../utilities')

import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay

if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    dSet = DataRetrieve()
    ct = ComputeTarget()
    
    issue = "tlt" 
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 2
    oos_months = 2
    segments = 2
    
    # get segmented dates
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]  
    is_end_date = is_start_date + relativedelta(months=is_months)
    oos_end_date = oos_start_date + relativedelta(months=oos_months)
    
    #load data
    dataSet = dSet.read_issue_data(issue)   
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, pivotDate)
    
    #set beLong level
    beLongThreshold = 0.0
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    
    for i in range(segments):
        modelData = dSet.set_date_range(dataSet,
                                        is_start_date,
                                        is_end_date
                                        )
        print ("IN SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("In Sample",
                            issue,
                            modelData,
                            is_start_date,
                            is_end_date
                            )
        is_start_date = is_end_date  + BDay(1)
        is_end_date = is_end_date + relativedelta(months=is_months) - BDay(1)
        
        # OOS
        modelData = dSet.set_date_range(dataSet,
                                        oos_start_date,
                                        oos_end_date
                                        )
        print ("OUT OF SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("Out of Sample",
                            issue,
                            modelData,
                            oos_start_date,
                            oos_end_date
                            )
        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=oos_months) - BDay(1)
        
        
        
        
        