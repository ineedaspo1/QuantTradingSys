# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

time_utils.py
"""
#import sys
#sys.path.append('../lib')
#sys.path.append('../utilities')


#from stat_tests import *

import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BDay

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

class TimeUtility:
    
    def is_oos_data_split(self, issue, pivotDate, isOosRatio, oosMonths, segments):
        return_dates = ()
        inSampleMonths = isOosRatio * oosMonths
        months_to_load = oosMonths + segments * inSampleMonths
        inSampleStartDate = pivotDate - relativedelta(months=months_to_load)
        oosStartDate = pivotDate - relativedelta(months=oosMonths * segments)
        dataLoadStartDate = inSampleStartDate - relativedelta(months=1)
        return_dates = (dataLoadStartDate,inSampleStartDate,oosStartDate,inSampleMonths)
        return return_dates

def print_beLongs(df):
    print ("beLong counts: ")
    print (df['beLong'].value_counts())
    print ("==========================")
    

if __name__ == "__main__":
    
    from plot_utils import *
    from retrieve_data import *
    from transformers import *
    timeUtil = TimeUtility()
    plotIt = PlotUtility()
    
    issue = "tlt" 
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 2
    oos_months = 2
    segments = 2
    
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    
    is_end_date = is_start_date + relativedelta(months=is_months)
    oos_end_date = oos_start_date + relativedelta(months=oos_months)
    
    print("inSampleMonths: " + str(is_months))
    print("Months to load: " + str(oos_months + segments * is_months))
    print("Load Date: ", dataLoadStartDate)
    print("In Sample Start  Date: ", is_start_date)
    print("Out of Sample Start Date: ", oos_start_date)
    print("Pivot Date: ", pivotDate)
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)   
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, pivotDate)
    
    #set beLong level
    beLongThreshold = 0.0
    ct = ComputeTarget()
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    
    for i in range(segments):
        modelData = dSet.set_date_range(dataSet, is_start_date, is_end_date)
        print ("IN SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("In Sample", issue, modelData, is_start_date, is_end_date)
        is_start_date = is_end_date  + BDay(1)
        is_end_date = is_end_date + relativedelta(months=is_months) - BDay(1)
        
        # OOS
        modelData = dSet.set_date_range(dataSet, oos_start_date, oos_end_date)
        print ("OUT OF SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("Out of Sample", issue, modelData, oos_start_date, oos_end_date)
        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=oos_months) - BDay(1)
        
        
        
        
        