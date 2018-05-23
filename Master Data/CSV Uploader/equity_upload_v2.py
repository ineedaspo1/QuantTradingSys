# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:44:29 2018

@author: kruegkj
"""

# equity_upload_v2.py

from datetime import datetime
import pandas_datareader.data as pdr
import pandas as pd
from pandas_datareader._utils import RemoteDataError
import os

def read_list_of_equities(csv_file_name):
    """
    Read in list of equity symbols from flat file located on OneDrive and 
    returns a df
    """
    download_path = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData', csv_file_name)
    content = pd.read_csv(download_path, delimiter=',', dtype='str')
    return content

def morningstar_data_loader(symbols):
    """
    Use list of symbols to retrieve price data from morningstar
    v1: Basic
    Upgrades: send date ranges
    """
    data_source = 'morningstar'
    start_date = datetime(1990,1,1)
    end_date = datetime.today()
    
    for issue in symbols:
        print ('===================================')
        print ('Issue: ' + issue)
    
        # Download data
        try:
            qt = pdr.DataReader(issue, data_source, start_date, end_date).reset_index()
        except RemoteDataError:
            print("No information for ticker '%s'" % issue)
            continue
        except TypeError:
            continue
        except:
            continue
        
        issue_name = issue + '.pkl'
        file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
        qt.to_pickle(file_name)

if __name__ == "__main__":
    csv_file_name = 'PotentialETFs.txt'
    symbols = read_list_of_equities(csv_file_name)
    #print(symbols)
    print("%s symbols were read." % len(symbols.columns))
    morningstar_data_loader(symbols)