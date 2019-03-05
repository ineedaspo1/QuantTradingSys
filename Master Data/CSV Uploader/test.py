# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:05:06 2019
LX4BIH9SO5EKE2SU
@author: kruegkj
"""

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import pandas_datareader.data as web
from datetime import datetime
import os
import pandas as pd
import numpy as np
from numpy import nan

##Get the data
#gotData = False 
#while gotData == False:
#		try:
#			ts = TimeSeries(key='LX4BIH9SO5EKE2SU', output_format = 'pandas')
#			amazon_data, amazon_meta_data = ts.get_daily_adjusted(symbol='AMZN', outputsize='full')
#			gotData = True
#		except:
#			print('alpha vantage api error')
#
#pprint(amazon_data.head(10))
#pprint(amazon_data.tail(10))

def read_list_of_equities(csv_file_name):
    """
    Read in list of equity symbols from flat file located on OneDrive and 
    returns a df
    """
    download_path = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData', csv_file_name)
    content = pd.read_csv(download_path, delimiter=',', dtype='str')
    return content


csv_file_name = 'PotentialETFs.txt'
symbols = read_list_of_equities(csv_file_name)
#print(symbols)
ts = TimeSeries(key='LX4BIH9SO5EKE2SU', output_format = 'pandas', indexing_type='integer')

test = 1
for symbol in symbols:
    print("symbol: ", symbol)
    try:
        dl_data, dl_meta = ts.get_daily_adjusted(symbol=symbol,
                                                 outputsize='full'
                                                 )
    except:
        print('alpha vantage api error')
        
    pprint(dl_data.head(10))
    
#    dl_data.rename(columns = {'$b':'B'}, inplace = True)
#    df.columns = df.columns.str.replace('$','')
    
    dl_data.drop(['7. dividend amount', '8. split coefficient'], axis=1, inplace=True)
    
    dl_data.rename(columns = {'date':'Date', 
                              '1. open':'Open',
                              '2. high': 'High',
                              '3. low': 'Low',
                              '4. close': 'Close',
                              '5. adjusted close': 'AdjClose',
                              '6. volume': 'Volume'}, inplace = True)
    
    #dl_data.loc[:, dl_data.columns != 'date'].rename(columns=lambda x: x[3:], inplace=True)
    pprint(dl_data.head(10))
    
    issue_name = symbol + '.pkl'
    file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
    dl_data.to_pickle(file_name)
    
    if test:
        break
    
