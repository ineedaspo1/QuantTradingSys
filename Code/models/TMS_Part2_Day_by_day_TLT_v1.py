# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj
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
import pickle
import random
import json
import sys
from scipy import stats

from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

if __name__ == '__main__':
       
    # set to existing system name OR set to blank if creating new
    if len(sys.argv) < 2:
        print('You must set a system_name or set to """"!!!')
    
    system_name = sys.argv[1]
    #ext_input_dict = sys.argv[2]
    
    print("Existing system")
    
    # Get info from system_dict
    system_directory = sysUtil.get_system_dir(system_name)
    if not os.path.exists(system_directory):
        print("system doesn't exist")
    else:
        file_name = 'system_dict.json'    
        system_dict = dSet.load_json(system_directory, file_name)
        issue = system_dict["issue"]
        direction = system_dict["direction"]
        ver_num = system_dict["ver_num"]
        # Perhaps only load these when needed?
        pivotDate = system_dict["pivotDate"]
        is_oos_ratio = system_dict["is_oos_ratio"]
        oos_months = system_dict["oos_months"]
        segments = system_dict["segments"]
    
    print(system_dict)
    
    # retrieve from CSV
    filename = "TMS_Part1_" + system_name + ".csv"
    path = system_directory+ "\\" + filename
    sst = pd.read_csv(path)
    
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
            sst1.iloc[i,sst1.columns.get_loc('fract')] = sst1.iloc[i,sst1.columns.get_loc('safef')]
            sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')]
        else:
            sst1.iloc[i,sst1.columns.get_loc('fract')] = sst1.iloc[i,sst1.columns.get_loc('safef')]
            sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')]
            
    for i in range(iStart, iEnd):
        if (sst1.signal[i] > 0):
            sst1.iloc[i,sst1.columns.get_loc('trade')] = sst1.iloc[i-1,sst1.columns.get_loc('fract')] * sst1.iloc[i-1,sst1.columns.get_loc('equity')] * sst1.iloc[i,sst1.columns.get_loc('gainAhead')]
        else:
            sst1.iloc[i,sst1.columns.get_loc('trade')] = 0.0
            
        sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')] + sst1.iloc[i,sst1.columns.get_loc('trade')]
        sst1.iloc[i,sst1.columns.get_loc('maxEquity')] = max(sst1.iloc[i,sst1.columns.get_loc('equity')],sst1.iloc[i-1,sst1.columns.get_loc('maxEquity')])
        sst1.iloc[i,sst1.columns.get_loc('drawdown')] = (sst1.iloc[i,sst1.columns.get_loc('maxEquity')] - sst1.iloc[i,sst1.columns.get_loc('equity')]) / sst1.iloc[i,sst1.columns.get_loc('maxEquity')]
        sst1.iloc[i,sst1.columns.get_loc('maxDD')] =  max(sst1.iloc[i,sst1.columns.get_loc('drawdown')],sst1.iloc[i-1,sst1.columns.get_loc('maxDD')])
        #print(sst1.iloc[i])
        #print('\n')
        
    # Show a few items
    print(sst1.tail(2))
    
    # Plot the equity curve and drawdown
    plotTitle = "Equity curve for  " + issue + ", " + str(start.strftime('%Y-%m-%d')) + " to " + str(end.strftime('%Y-%m-%d'))
    plotIt.plot_v1(sst1['equity'][:-2], plotTitle)
    plotTitle = "Drawdown for  " + issue + ", " + str(start.strftime('%Y-%m-%d')) + " to " + str(end.strftime('%Y-%m-%d'))
    plotIt.plot_v1(sst1['drawdown'][:-2], plotTitle)
    
    df_to_save = sst1.copy()
    df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    filename = "TMS_Part2_daybyday_" + system_name + ".csv"
    df_to_save.to_csv(system_directory+ "\\" + filename, encoding='utf-8', index=False)
    
    # Plot code
    fig = plt.figure(figsize=(11,6))
    fig.suptitle('CAR25 and issue price for' + issue)
    ax1 = fig.add_subplot(111)
    #ax1.plot(sst1.safef, color='green',label='safe-f')
    ax1.plot(sst1.CAR25, color='blue',label='CAR25')
    #ax1.plot(valData.equityValBeLongSignals, color='purple',label='ValBeLong')
    
    ax1.legend(loc='upper left', frameon=True, fontsize=8)
    ax1.label_outer()
    ax1.tick_params(axis='x',which='major',bottom=True)
    ax1.minorticks_on()
    ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
    ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
    
    #sst1['Pri']=valData.Pri
    ax2 = ax1.twinx()
    ax2.plot(sst1.Close, color='black',alpha=0.6,label='CLOSE',linestyle='--')
    ax2.legend(loc='center left', frameon=True, fontsize=8)
    ax2.label_outer()
    fig.autofmt_xdate()
    
    plotTitle = "Safe-f for " + issue + ", " + str(start.strftime('%Y-%m-%d')) + " to " + str(end.strftime('%Y-%m-%d'))
    plotIt.plot_v1(sst1['safef'][:-2], plotTitle)