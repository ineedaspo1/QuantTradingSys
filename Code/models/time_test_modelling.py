# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

time_test_modelling.py
"""
import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import *
from time_utils import *
from retrieve_data import *

import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt

from pandas.tseries.offsets import BDay


def print_beLongs(df):
    print ("beLong counts: ")
    print (df['beLong'].value_counts())
    print ("==========================") 

if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    
    issue = "tlt" 
    pivotDate = datetime.date(2018, 4, 2)
    inSampleOutOfSampleRatio = 2
    outOfSampleMonths = 2
    segments = 2
    
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, inSampleOutOfSampleRatio, outOfSampleMonths, segments)
    
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    inSampleMonths = isOosDates[3]  
    is_end_date = is_start_date + relativedelta(months=inSampleMonths)
    oos_end_date = oos_start_date + relativedelta(months=outOfSampleMonths)
    
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
        is_end_date = is_end_date + relativedelta(months=inSampleMonths) - BDay(1)
        
        # OOS
        modelData = dSet.set_date_range(dataSet, oos_start_date, oos_end_date)
        print ("OUT OF SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("Out of Sample", issue, modelData, oos_start_date, oos_end_date)
        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=outOfSampleMonths) - BDay(1)
        
        
        
        
        