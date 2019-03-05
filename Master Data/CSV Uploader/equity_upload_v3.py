# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:58:53 2019

@author: kruegkj
"""

import csv
import time
import urllib
import json
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries



MAX_SYMBOLS = 500
MAX_REQUESTS_PER_MINUTE = 5
SLEEP_TIME = 60 / MAX_REQUESTS_PER_MINUTE

def read_list_of_equities(csv_file_name):
    """
    Read in list of equity symbols from flat file located on OneDrive and 
    returns a df
    """
    download_path = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData', csv_file_name)
    content = pd.read_csv(download_path, delimiter=',', dtype='str')
    return content

def get_quotes(symbols:list):
    """
    Get quotes for each symbol.
    """
    ts = TimeSeries(key='LX4BIH9SO5EKE2SU', output_format = 'pandas')
    gotData = False
    while gotData == False:
        
		try:
			amazon_data, amazon_meta_data = ts.get_daily_adjusted(symbol='AMZN', outputsize='full')
			gotData = True
		except:
			print('alpha vantage api error')
    iteration_count = 0

    for symbol in symbols:

        try:
            iteration_count += 1
            progress_bar.update(iteration_count, symbol)
            my_url = url.format(symbol)
            response = urllib.request.urlopen(my_url)
            csv_data = response.read()

            # If the first 1,000 characters does not contain something that looks like an error message,
            # then save the data.
            if csv_data[:1000].find(b"Error Message") == -1 and csv_data[:1000].find(b'"Note":') == -1:
                raw_prices.save(symbol, csv_data)
                time.sleep(SLEEP_TIME)
        except urllib.error.HTTPError:
            LOGGER.error("HTTP Error retrieving data for %s", symbol)
        except urllib.error.URLError:
            LOGGER.error("URL Error retrieving data for %s from %s", symbol, my_url)
        except Exception as e:
            LOGGER.error("Error retrieving data for %s from %s", symbol, my_url)
            LOGGER.exception(e)
            
def main():
    """
    Main processing function.
    """
    csv_file_name = 'PotentialETFs.txt'
    symbols = read_list_of_equities(csv_file_name)
    print(symbols)
#    symbol_lister = SymbolLister(MAX_SYMBOLS)
#    if symbol_lister.symbol_limit > 0:
#        print("Will only download the first {} symbols.".format(symbol_lister.symbol_limit))
#    symbols = symbol_lister.get_symbols()
    get_quotes(symbols)

if __name__ == "__main__":
    main()