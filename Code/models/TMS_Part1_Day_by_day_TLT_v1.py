# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj
TMS Part 1 day by day
"""

# Import standard libraries
import pandas as pd
#import numpy as np
#import datetime as dt
#from dateutil.relativedelta import relativedelta
#import matplotlib.pylab as plt
#from pandas.tseries.offsets import BDay
#import os
import sys

# Import custom libraries
from Code.lib.plot_utils import PlotUtility
#from Code.lib.time_utils import TimeUtility
from Code.lib.retrieve_data import DataRetrieve
from Code.lib.retrieve_system_info import TradingSystemUtility
#from Code.lib.candle_indicators import CandleIndicators
#from Code.lib.transformers import Transformers
#from Code.lib.ta_momentum_studies import TALibMomentumStudies
#from Code.lib.model_utils import ModelUtility, TimeSeriesSplitImproved
#from Code.lib.feature_generator import FeatureGenerator
#from Code.utilities.stat_tests import stationarity_tests
#from Code.lib.config import current_feature, feature_dict
#from Code.models import models_utils
#from Code.lib.model_algos import AlgoUtility
from Code.lib.tms_utils import TradeRisk

plotIt = PlotUtility()
#timeUtil = TimeUtility()
#ct = ComputeTarget()
#taLibMomSt = TALibMomentumStudies()
#transf = Transformers()
#modelUtil = ModelUtility()
#featureGen = FeatureGenerator()
dSet = DataRetrieve()
#modelAlgo = AlgoUtility()
sysUtil = TradingSystemUtility()
tradeRisk = TradeRisk()

if __name__ == '__main__':
       
    # set to existing system name OR set to blank if creating new
    if len(sys.argv) < 2:
        print('You must set a system_name or set to """"!!!')
    
    system_name = sys.argv[1]
    system_directory = sysUtil.get_system_dir(system_name)
    #ext_input_dict = sys.argv[2]
    
    print("Existing system")
    
    # Get info from system_dict
    system_dict = sysUtil.get_dict(system_directory, 'system_dict')
    issue = system_dict["issue"]
    
    is_oos_ratio = system_dict["is_oos_ratio"]
    oos_months = system_dict["oos_months"]
    segments = system_dict["segments"]
    
    print(system_dict)
    
    sst1 = dSet.read_csv(system_directory,
                        system_name,
                        'OOS_Equity',
                        'dbd'
                        )
    
    sst = sst1.copy()
    
    # Initialize dataframe
    sst = sst.set_index(pd.DatetimeIndex(sst['Date']))
    sst=sst.drop('Date', axis=1)
    sst['safef'] = 0.0
    sst['CAR25'] = 0.0
    sst.head(2)
    
    # Get start and end date of data frame
    print(sst.index[0])
    print(sst.index[-1])
    
    start = sst.index[0]
    end = sst.index[-1]
    updateInterval = 1
    
    # Initialize analysis variables
    f_days = sst.shape[0]
    print(f_days)
    
    forecastHorizon = f_days*2
    initialEquity = 10000
    ddTolerance = 0.10
    tailRiskPct = 95
    windowLength = int(0.25*forecastHorizon)
    nCurves = 50
    
    years_in_forecast = forecastHorizon / 252.0
    print("Years in forecast: " + str(years_in_forecast))
    print("Window length: " + str(windowLength))
    
    # Create tms_dict
    tms_dict = {}
    tms_dict = {'forecastHorizon' : forecastHorizon,
               'initialEquity'    : initialEquity,
               'ddTolerance'      : ddTolerance,
               'tailRiskPct'      : tailRiskPct,
               'windowLength'     : windowLength,
               'nCurves'          : nCurves,
               'updateInterval'   : updateInterval
               }
    
    sysUtil.save_dict(system_name, 'tms_dict', tms_dict)
    
    # Work with index instead of dates
    start = sst.index[0+windowLength]
    iStart = sst.index.get_loc(start)
    print(iStart)
    iEnd = sst.index.get_loc(end)
    print(iEnd)
    
    printDetails = False
    
    tradeRisk.get_safef_car25(sst, iStart, iEnd, tms_dict)
        
    print(sst.tail(3)) 
    sst.reset_index(level='Date', inplace=True)
    print(sst.tail(3))
#    dSet.save_csv(system_directory,
#                  system_name,
#                  'TMS_Part1',
#                  'dbd',
#                  sst
#                  )
    
    plot_tms = sst.set_index(pd.DatetimeIndex(sst1['Date']))
    plot_tms=plot_tms.drop('Date', axis=1)
    plotTitle = "Safe-f for " + issue
    plotIt.plot_v1(plot_tms['safef'][:-2], plotTitle)
    plotTitle = "CAR25 for " + issue
    plotIt.plot_v1(plot_tms['CAR25'][:-2], plotTitle)
