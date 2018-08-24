# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:44:29 2018

@author: kruegkj
file: auxiliary_upload.py
"""

# equity_upload_v2.py

from datetime import datetime
import pandas_datareader.data as pdr
import pandas as pd
from pandas_datareader._utils import RemoteDataError
import os
import pickle

def read_list_of_aux_data(csv_file_name):
    """
    Read in list of equity symbols from flat file located on OneDrive and 
    returns a df
    """
    download_path = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData', csv_file_name)
    content = pd.read_csv(download_path, delimiter=',', dtype='str')
    return content

        
def fred_data_loader(symbols):
    data_source = 'fred'
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
        file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Auxiliary', issue_name)
        pickle.dump(qt, open(file_name, "wb" ))
    

if __name__ == "__main__":
    # FRED symbols here: https://fred.stlouisfed.org/categories
    csv_file_name = 'AuxiliarySymbols.txt'
    symbols = read_list_of_aux_data(csv_file_name)
    #print(symbols)
    print("%s symbols were read." % len(symbols.columns))
    fred_data_loader(symbols)
    
    
    
    
    
    