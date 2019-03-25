# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj
TMS Part 1 day by day
"""

# Import standard libraries
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os
import os.path
import sys
from scipy import stats

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
    
    forecastHorizon = f_days
    initialEquity = 10000
    ddTolerance = 0.10
    tailRiskPct = 95
    windowLength = .4*forecastHorizon
    nCurves = 50
    
    years_in_forecast = forecastHorizon / 252.0
    print(years_in_forecast)
    
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
    iStart = sst.index.get_loc(start)
    print(iStart)
    iEnd = sst.index.get_loc(end)
    print(iEnd)
    
    printDetails = False
    
    for i in range(iStart, iEnd+1, updateInterval):
        print ("\nDate: ", dt.datetime.strftime(sst.index[i], '%Y-%m-%d'))
        print ("beLong: ", sst.signal[i])
        print ("gain Ahead: {0:.4f}".format(sst.gainAhead[i]))
    
    #  Initialize variables
        curves = np.zeros(nCurves)
        numberDraws = np.zeros(nCurves)
        TWR = np.zeros(nCurves)
        maxDD = np.zeros(nCurves)
        
        fraction = 1.00
        dd95 = 2 * ddTolerance
        
        while (abs(dd95-ddTolerance)>0.03):
            #  Generate nCurve equity curves
            print  ("    Fraction {0:.2f}".format(fraction))
    #    
            for nc in range(nCurves):
                #print ("working on curve ", nc)
                equity = initialEquity
                maxEquity = equity
                drawdown = 0
                maxDrawdown = 0
                horizonSoFar = 0
                nd = 0
                while (horizonSoFar < forecastHorizon):
                    j = np.random.randint(0,windowLength)
                    #print(j)
                    nd = nd + 1
                    weightJ = 1.00 - j/windowLength
                    #print (weightJ)
                    horizonSoFar = horizonSoFar + weightJ
                    signalJ = sst.signal[i-j]
                    if signalJ > 0:
                        tradeJ = sst.gainAhead[i-j] * weightJ
                    else:
                        tradeJ = 0.0
                    thisTrade = fraction * tradeJ * equity    
                    equity = equity + thisTrade
                    maxEquity = max(equity,maxEquity)
                    drawdown = (maxEquity-equity)/maxEquity
                    maxDrawdown = max(drawdown,maxDrawdown)
                #print ("equity, maxDD, ndraws:", equity, maxDrawdown, nd)        
                TWR[nc] = equity
                maxDD[nc] = maxDrawdown
                numberDraws[nc] = nd
        
            #  Find the drawdown at the tailLimit-th percentile        
            dd95 = stats.scoreatpercentile(maxDD,tailRiskPct)
            #print ("  DD %i: %.3f " % (tailRiskPct, dd95))
            fraction = fraction * ddTolerance / dd95
            TWR25 = stats.scoreatpercentile(TWR,25)        
            CAR25 = 100*(((TWR25/initialEquity) ** (1.0/years_in_forecast))-1.0)
        
        print ('Fraction: {0:.2f}'.format(fraction))
        print ('CAR25: {0:.2f}'.format(CAR25))
        sst.iloc[i,sst.columns.get_loc('safef')] = fraction
        sst.iloc[i,sst.columns.get_loc('CAR25')] = CAR25
        #sst.loc[i,'CAR25'] = CAR25
        
    print(sst.tail(3)) 
    sst.reset_index(level='Date', inplace=True)
    print(sst.tail(3))   
#    dSet.save_csv(system_directory,
#                  system_name,
#                  'TMS_Part1',
#                  'dbd',
#                  sst
#                  )