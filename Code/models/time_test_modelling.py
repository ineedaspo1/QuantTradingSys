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
    return_dates = []
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
    return_dates.append(dataLoadStartDate)
    return_dates.append(inSampleStartDate)
    return_dates.append(oosStartDate)
    return_dates.append(inSampleMonths)
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
    oos_end_date = oosStartDate + relativedelta(months=outOfSampleMonths)
    
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
    
    # Put indicators and transforms here
    
    #set beLong level
    beLongThreshold = 0.0
    ct = ComputeTarget()

    # IS only
    for i in range(segments):
        df2 = pd.date_range(start=is_start_date, end=is_end_date, freq=us_cal)
        print ("IS Start Date: ", is_start_date)
        print ("IS End Date: ", is_end_date)

        modelData = dataSet.reindex(df2)
        
        # set target var
        mmData = ct.setTarget(modelData, "Long", beLongThreshold)
        print_beLongs(mmData)
        
        mmData = mmData.drop(['Open','High','Low','Close'],axis=1)
        
        plot_beLongs("In Sample", issue, mmData, is_start_date, is_end_date)
             
        is_start_date = is_end_date  + BDay(1)
        is_end_date = is_end_date + relativedelta(months=inSampleMonths) - BDay(1)
        
        # OOS
        df3 = pd.date_range(start=oos_start_date, end=oos_end_date, freq=us_cal)
        print ("OOS Start Date: ", oos_start_date)
        print ("OOS End Date: ", oos_end_date)

        modelData = dataSet.reindex(df3)
        # set target var
        mmData = ct.setTarget(modelData, "Long", beLongThreshold)
        print_beLongs(mmData)
        
        mmData = mmData.drop(['Open','High','Low','Close'],axis=1)
        plot_beLongs("Out of Sample", issue, mmData, oos_start_date, oos_end_date)

        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=outOfSampleMonths) - BDay(1)
        
        
        
        
        
        
        
        