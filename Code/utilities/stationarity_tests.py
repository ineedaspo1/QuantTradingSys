# -*- coding: utf-8 -*-
"""
Created on Sat May 12 06:13:51 2018

@author: kruegkj

stationarity_tests.py
"""
import sys
sys.path.append('../lib')

# Import the Time Series library
import statsmodels.tsa.stattools as ts
from retrieve_issue_data import read_issue_data
from stat_tests import *
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 14,3


if __name__ == "__main__":
    issue = "xle"
    
    years_to_load = 4
    segments = 8
    segment_months = int((years_to_load*12)/segments)
    
    # date parsing for analysis
    dataLoadEndDate = datetime.date(2018, 3, 30) 
    dataLoadStartDate = dataLoadEndDate - relativedelta(years=years_to_load)
    print(dataLoadStartDate)
    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    
    # increment through time segments
    i = 0
    startPlotDate = dataLoadStartDate
    while i < segments:
        endPlotDate = startPlotDate + relativedelta(months=segment_months)
        qtPlot = dataSet.ix[startPlotDate:endPlotDate]      
        orig = plt.plot(qtPlot.Close, color='blue', label=issue)
        plt.legend(loc='best')
        print("\n\n\n")
        plt.show(block=False)
        startPlotDate = endPlotDate + relativedelta(days=1)
        i+=1
      
        adf_test(qtPlot,issue)
        hurst_setup(qtPlot['Close'][:],issue)

