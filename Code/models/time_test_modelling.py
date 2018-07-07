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
from retrieve_data import *
from indicators import *
from transformers import *
#from stat_tests import *

# Import the Time Series library
import statsmodels.tsa.stattools as ts
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
plt.style.use('seaborn-ticks')
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BDay
import matplotlib as mpl
plt.style.use('seaborn-ticks')
import matplotlib.ticker as ticker

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

plotIt = PlotUtility()

def is_oos_data_split(issue, pivotDate, isOosRatio, oosMonths, segments):
    return_dates = ()
    inSampleMonths = isOosRatio * oosMonths
    print("inSampleMonths: " + str(inSampleMonths))
    months_to_load = oosMonths + segments * inSampleMonths
    print("Months to load: " + str(months_to_load))
    inSampleStartDate = pivotDate - relativedelta(months=months_to_load)
    oosStartDate = pivotDate - relativedelta(months=oosMonths * segments)
    dataLoadStartDate = inSampleStartDate - relativedelta(months=1)
    print("Load Date: ", dataLoadStartDate)
    print("In Sample Start  Date: ", inSampleStartDate)
    print("Out of Sample Start Date: ", oosStartDate)
    print("Pivot Date: ", pivotDate)
    return_dates = (dataLoadStartDate,inSampleStartDate,oosStartDate,inSampleMonths)
    return return_dates

def print_beLongs(df):
    print ("beLong counts: ")
    print (df['beLong'].value_counts())
    print ("out of ", nrows)
    print ("==========================")

def plot_beLongs(title, issue, df, start_date, end_date):
    plotTitle = title + ": " + issue + ", " + str(start_date) + " to " + str(end_date)
    plotIt.plot_v2x(df['Pri'], df['beLong'], plotTitle)
    plotIt.histogram(df['beLong'], x_label="beLong signal", y_label="Frequency", title = "beLong distribution for " + issue)
    
def trim_dates(start_date, end_date, df):
    df2 = pd.date_range(start=start_date, end=end_date, freq=us_cal)
    print ("Start Date: ", start_date)
    print ("End Date: ", end_date)
    modelData = df.reindex(df2)
    return modelData

if __name__ == "__main__":
    issue = "tlt" 
    pivotDate = datetime.date(2018, 4, 2)
    inSampleOutOfSampleRatio = 2
    outOfSampleMonths = 2
    segments = 2
    
    isOosDates = is_oos_data_split(issue, pivotDate, inSampleOutOfSampleRatio, outOfSampleMonths, segments)
    
    dataLoadStartDate = isOosDates[0]
    inSampleStartDate = isOosDates[1]
    oos_start_date = isOosDates[2]
    inSampleMonths = isOosDates[3]
    
    is_start_date = inSampleStartDate
    is_end_date = is_start_date + relativedelta(months=inSampleMonths)
    oos_end_date = oos_start_date + relativedelta(months=outOfSampleMonths)
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)   
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, pivotDate)
    
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    print ("================")
    
    # set lag on Close (Pri)
    transf = Transformers()
    lag_var = 'Pri'
    lags = 5
    dataSet = transf.add_lag(dataSet, lag_var, lags)
    
    # set % return variables and lags
    dataSet["percReturn"] = dataSet["Pri"].pct_change()*100
    lag_var = 'percReturn'
    lags = 5    
    dataSet = transf.add_lag(dataSet, lag_var, lags)    
    
    #set beLong level
    beLongThreshold = 0.0
    ct = ComputeTarget()
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    
    for i in range(segments):
        modelData = trim_dates(is_start_date, is_end_date, dataSet)
        print ("IN SAMPLE")
        print_beLongs(modelData)
        plot_beLongs("In Sample", issue, modelData, is_start_date, is_end_date)
        is_start_date = is_end_date  + BDay(1)
        is_end_date = is_end_date + relativedelta(months=inSampleMonths) - BDay(1)
        
        # OOS
        modelData = trim_dates(oos_start_date, oos_end_date, dataSet)
        print ("OUT OF SAMPLE")
        print_beLongs(modelData)
        plot_beLongs("Out of Sample", issue, modelData, oos_start_date, oos_end_date)
        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=outOfSampleMonths) - BDay(1)
        
        
        
        
        