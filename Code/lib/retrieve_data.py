# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:17:54 2018
@author: kruegkj
retrieve_data.py
"""
import pandas as pd
import numpy as np
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pickle

import datetime as dt

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

class DataRetrieve:   
    """Class of functions used to retrieve issue data"""
    def read_pickle_data(self, file_name, issue):
        """Reads data from a pickle file   
           Args:
                file_name: filename of pickle file
                issue: symbol of data to be retrieved
           Return:
                df: Dataframe of issue data
        """
        try:
            df = pd.read_pickle(file_name)
        except:
            print("================================")
            print("No information for ticker '%s'" % issue)
            print("================================")
            raise SystemExit
        print ("Successfully retrieved data series for " + issue)
        return df
        
    def read_issue_data(self, issue):
        """Reads equity from a pickle file   
           Args:
                issue: symbol of data to be retrieved
           Return:
                df: Dataframe of issue data
           To update: Set location of filename location outside of class
        """
        self.issue = issue
        issue_name = issue + '.pkl'
        file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
        df = self.read_pickle_data(file_name, issue)
        return df
    
    def read_fred_data(self, issue):
        """Reads FRED data from a pickle file   
           Args:
                issue: symbol of data to be retrieved
           Return:
                df: Dataframe of issue data
           To update: Set location of filename location outside of class
        """
        self.issue = issue
        issue_name = issue + '.pkl'
        file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Auxiliary', issue_name)
        print(file_name)
        df = self.read_pickle_data(file_name, issue)
        return df
        
    def set_date_range(self, df, dfStartDt, dfEndDt, dateName='Date'):
        """Set US bus cal date range to be retrieved from datetimeindex   
           Args:
                df: dataframe of issue data
                dfStartDt: start date
                dfEndDt: end date
                dateName: name of Date column, default is 'Name'
           Return:
                df3: Dataframe of issue data
           To update: Set location of filename location outside of class
        """
        us_cal = CustomBusinessDay(calendar=USTradingCalendar())
        df.set_index(pd.to_datetime(df[dateName]), inplace=True)
        df3 = df.reindex(pd.date_range(start=dfStartDt, end=dfEndDt, freq=us_cal))
        return df3

    def drop_columns(self, df, col_vals):
        """Utility to drop columns   
           Args:
                df: dataframe of issue data
                col_vals: list of column names
           Return:
                df: Dataframe with columns dropped
           To update: Verify columns exist before dropping
        """
        df.drop(col_vals, axis =1, inplace=True)
        return df
    
    def save_obj(self, obj, name ):
        with open('../obj/'+ name + '.pkl', 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    def load_obj(self, name):
        with open('../obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    
class ComputeTarget:
    
    def setTarget(self, df, direction, beLongThreshold):
        """Set value of target to 1 if gainAhead>0, otherwise set to 0   
           Args:
                df: dataframe of issue data
                direction: long or short (currently unused, assume long)
                beLongThreshold: gain must be above this threshold to 
                    set direction
           Return:
                df: Dataframe with gainAhead and beLong columns added
           To update: Added direction and correct code to identify
               long and short thresholds and correct signal
        """
        df['gainAhead'] = ComputeTarget.gainAhead(df.Close)
        df['beLong'] = np.where(df.gainAhead>beLongThreshold,1,-1)
        return df

    def gainAhead(p):
        """Computes change in the next 1 bar. A change of 1% is 0.01.
           The final value is unknown.  Its value is 0.0. 
           Args:
                p: price series
           Return:
                g: numpy array of changes
        """
        nrows = p.shape[0]
        g = np.zeros(nrows)
        for i in range(0,nrows-1):
            g[i] = (p[i+1]-p[i])/p[i]
            # if % change is 0, change to small number
            if (abs(g[i]) < 0.0001):
                g[i] = 0.0001
        return g
        
    def priceChange(self, p):
        """Computes pricechange in the next 1 bar. 
           The final value is unknown.  Its value is 0.0. 
           Args:
                p: price series
           Return:
                pc: numpy array of changes
        """
        nrows = p.shape[0]
        pc = np.zeros(nrows)
        for i in range(1,nrows):
            pc[i] = (p[i]-p[i-1])/p[i-1]
        return pc
        
if __name__ == "__main__":
    from plot_utils import *
    plotIt = PlotUtility()
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-06-01"
    issue = "TLT"
    aux_issue = "VIXCLS"
    threeMoTbill = "DTB3"
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(
            dataSet, 
            dataLoadStartDate,
            dataLoadEndDate)
    
    vixDataSet = dSet.read_fred_data(aux_issue)
    vixDataSet = dSet.set_date_range(
            vixDataSet, 
            dataLoadStartDate,
            dataLoadEndDate,
            dateName="DATE")
    
    threeMoDataSet = dSet.read_fred_data(threeMoTbill)
    threeMoDataSet = dSet.set_date_range(
            threeMoDataSet, 
            dataLoadStartDate,
            dataLoadEndDate,
            dateName="DATE")
    
    beLongThreshold = 0.0
    ct = ComputeTarget()
    targetDataSet = ct.setTarget(
            dataSet, 
            "Long", 
            beLongThreshold)
    nrows = targetDataSet.shape[0]
    print ("nrows: ", nrows)
    print (targetDataSet.shape)
    print (targetDataSet.tail(10))
    
    #targetDataSet = dSet.drop_columns(targetDataSet,['High','Low'])
    
    print ("beLong counts: ")
    print (targetDataSet['beLong'].value_counts())
    print ("out of ", nrows)
    
    testFirstYear = "2014-04-01"
    testFinalYear = "2014-06-01"
    qtPlot = targetDataSet.ix[testFirstYear:testFinalYear]
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(qtPlot['Close'], plotTitle)
    
    plotTitle = issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v2x(qtPlot['Close'], qtPlot['beLong'], plotTitle)
    
    plotTitle = "VIX Closing"
    plotIt.plot_v1(vixDataSet['VIXCLS'], plotTitle)
    
    plotTitle = "3 month TBill"
    plotIt.plot_v1(threeMoDataSet['DTB3'], plotTitle)
    
    # Merged dataSet confirmation
    print(dataSet.Close.head(20))
    print(vixDataSet.head(20))
    merged_result = dataSet.join(vixDataSet, how='outer')
    print(merged_result.head(20))
    
    plotTitle = "VIX and " + str(issue) + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v2x(merged_result['Close'], merged_result['VIXCLS'], plotTitle)