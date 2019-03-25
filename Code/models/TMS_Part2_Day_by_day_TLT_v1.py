# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj
TMS Part 2 day by day
"""

# Import standard libraries
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os
import os.path
import sys

# Import custom libraries
from Code.lib.plot_utils import PlotUtility
from Code.lib.time_utils import TimeUtility
from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
from Code.lib.retrieve_system_info import TradingSystemUtility
from Code.lib.candle_indicators import CandleIndicators
from Code.lib.transformers import Transformers
from Code.lib.ta_momentum_studies import TALibMomentumStudies
from Code.lib.model_utils import ModelUtility, TimeSeriesSplitImproved
from Code.lib.feature_generator import FeatureGenerator
from Code.utilities.stat_tests import stationarity_tests
from Code.lib.config import current_feature, feature_dict
from Code.models import models_utils
from Code.lib.model_algos import AlgoUtility
from Code.lib.tms_utils import TradeRisk

plotIt = PlotUtility()
timeUtil = TimeUtility()
ct = ComputeTarget()
candle_ind = CandleIndicators()
dSet = DataRetrieve()
taLibMomSt = TALibMomentumStudies()
transf = Transformers()
modelUtil = ModelUtility()
featureGen = FeatureGenerator()
dSet = DataRetrieve()
modelAlgo = AlgoUtility()
sysUtil = TradingSystemUtility()
tradeRisk = TradeRisk()

if __name__ == '__main__':
    from Code.lib.plot_utils import PlotUtility
    plotIt = PlotUtility()
       
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
    
    # Load TMS Part 1
    sst = dSet.read_csv(system_directory,
                         system_name,
                         'TMS_Part1',
                         'dbd'
                         )
    
    sst = sst.set_index(pd.DatetimeIndex(sst['Date']))
    sst=sst.drop('Date', axis=1)
    print(sst.tail(5))

    # Compute equity, maximum equity, drawdown, and maximum drawdown
    # Init vars
    sst1 = sst.copy()
    sst1['trade'] = 0.0
    sst1['fract'] = 0.0
    sst1['equity'] = 0.0
    sst1['maxEquity'] = 0.0
    sst1['drawdown'] = 0.0
    sst1['maxDD'] = 0.0
    sst1['trade_decision'] = ''
    initialEquity = 100000
    
    nrows = sst1.shape[0]
    start = sst.index[0]
    end = sst.index[-1]
    
    iStart = sst.index.get_loc(start)
    iEnd = sst.index.get_loc(end)
    
    sst1.iloc[0,sst1.columns.get_loc('safef')] = 1.0
    sst1.iloc[0,sst1.columns.get_loc('CAR25')] = 10.0
    sst1.iloc[0,sst1.columns.get_loc('equity')] = initialEquity
    
    for i in range(1,nrows):
        if (sst1.iloc[i,sst1.columns.get_loc('safef')]==0 and sst1.iloc[i,sst1.columns.get_loc('CAR25')]==0):
            sst1.iloc[i,sst1.columns.get_loc('safef')] = sst1.iloc[i-1,sst1.columns.get_loc('safef')]
            sst1.iloc[i,sst1.columns.get_loc('CAR25')] = sst1.iloc[i-1,sst1.columns.get_loc('CAR25')]
            #sst1.CAR25[i] = sst1.CAR25[i-1]
            sst1.iloc[i,sst1.columns.get_loc('fract')] = sst1.iloc[i,sst1.columns.get_loc('safef')]
            #sst1.fract[i] = sst1.safef[i]
            sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')]
        else:
            #sst1.fract[i] = sst1.safef[i]
            sst1.iloc[i,sst1.columns.get_loc('fract')] = sst1.iloc[i,sst1.columns.get_loc('safef')]
            #st1.equity[i] = sst1.equity[i-1]
            sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')]
            
    for i in range(iStart, iEnd):
        if (sst1.signal[i] > 0):
            #sst1.trade[i] = sst1.fract[i-1] * sst1.equity[i-1] * sst1.gainAhead[i]
            sst1.iloc[i,sst1.columns.get_loc('trade')] = sst1.iloc[i-1,sst1.columns.get_loc('fract')] * sst1.iloc[i-1,sst1.columns.get_loc('equity')] * sst1.iloc[i,sst1.columns.get_loc('gainAhead')]
        else:
            #sst1.trade[i] = 0.0
            sst1.iloc[i,sst1.columns.get_loc('trade')] = 0.0
        #sst1.equity[i] = sst1.equity[i-1] + sst1.trade[i]
        sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')] + sst1.iloc[i,sst1.columns.get_loc('trade')]
        #sst1.maxEquity[i] = max(sst1.equity[i],sst1.maxEquity[i-1])
        sst1.iloc[i,sst1.columns.get_loc('maxEquity')] = max(sst1.iloc[i,sst1.columns.get_loc('equity')],sst1.iloc[i-1,sst1.columns.get_loc('maxEquity')])
        #sst1.drawdown[i] = (sst1.maxEquity[i]-sst1.equity[i]) / sst1.maxEquity[i]
        sst1.iloc[i,sst1.columns.get_loc('drawdown')] = (sst1.iloc[i,sst1.columns.get_loc('maxEquity')] - sst1.iloc[i,sst1.columns.get_loc('equity')]) / sst1.iloc[i,sst1.columns.get_loc('maxEquity')]
        #sst1.maxDD[i] = max(sst1.drawdown[i],sst1.maxDD[i-1])
        sst1.iloc[i,sst1.columns.get_loc('maxDD')] =  max(sst1.iloc[i,sst1.columns.get_loc('drawdown')],sst1.iloc[i-1,sst1.columns.get_loc('maxDD')])

        
    # Show a few items
    print(sst1.tail(2))
    sst1.reset_index(level='Date', inplace=True)
    
    dSet.save_csv(system_directory,
                  system_name,
                  'TMS_Part2',
                  'dbd',
                  sst1
                  )
    
    plot_tms = sst1.set_index(pd.DatetimeIndex(sst1['Date']))
    plot_tms=plot_tms.drop('Date', axis=1)
    
    # Plot the equity curve and drawdown
    plotIt.plot_equity_drawdown(issue, plot_tms)
    
#    plot_tms = sst1.set_index(pd.DatetimeIndex(sst1['Date']))
#    plot_tms=plot_tms.drop('Date', axis=1)
#    plotTitle = "Equity curve for  " + issue
#    plotIt.plot_v1(plot_tms['equity'][:-2], plotTitle)
#    plotTitle = "Drawdown for  " + issue
#    plotIt.plot_v1(plot_tms['drawdown'][:-2], plotTitle)
#    plt.show()
    
    plotIt.plot_CAR25_close(issue, plot_tms)
    
    plotTitle = "Safe-f for " + issue
    plotIt.plot_v1(plot_tms['safef'][:-2], plotTitle)