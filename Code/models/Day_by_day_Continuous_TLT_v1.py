# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj

Day-by-day continuous
"""

# Import standard libraries
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os
import os.path
import pickle
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
    
    # Set IS-OOS parameters
    pivotDate = system_dict["pivotDate"]
    pivotDate = datetime.strptime(pivotDate, '%Y-%m-%d')
    print('pivotDate: ', pivotDate)
    pivotDate = datetime.date(pivotDate)
    
    # set date splits
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]
    
    modelStartDate = oos_start_date
    
    # Retrieve feature_dict
    feature_dict = sysUtil.get_dict(system_directory, 'feature_dict')
    
    # get raw data 
    raw_df = dSet.read_issue_data(issue)
    
    # Set trade date
    from datetime import timedelta
    tradeDate = pivotDate + BDay(1)
    print('tradeDate: ', tradeDate)

    if (tradeDate.strftime('%Y-%m-%d') in raw_df['Date'].values):
        print ('date exist')
    else:
        print('no date')
        tradeDate += BDay(1)
        print('tradeDate: ', tradeDate)
        
    # set trade date
    tradeDate = tradeDate.strftime('%Y-%m-%d')
    print(tradeDate)
    
    # set df to trade date range
    trimmed_df = raw_df.iloc[0:raw_df[raw_df.Date == tradeDate].index[0]]
    print(trimmed_df.tail(2))

    price_loc = raw_df.index[raw_df.Date == tradeDate]
    index = price_loc[0]
    print(index)

    # get Close from trade date
    new_open = raw_df.Close[index]
    print(new_open)

    # append price to df
    new_data_df = trimmed_df.append({'Date' : tradeDate , 'Open' : raw_df.Open[index], 'High' : raw_df.High[index], 'Low' : raw_df.Low[index], 'Close' : raw_df.Close[index], 'AdjClose' : raw_df.AdjClose[index], 'Volume' : raw_df.Volume[index] } , ignore_index=True)
    print(new_data_df.tail(2))

    # get first row
    df_start_date = new_data_df.Date[0]
    
    # get last row
    lastRow = new_data_df.shape[0]
    df_end_date = new_data_df.Date[lastRow-1]
    
    feat_df = dSet.set_date_range(new_data_df, df_start_date, df_end_date)
    # Resolve any NA's for now
    feat_df.fillna(method='ffill', inplace=True)
    
    #set beLong level
    beLongThreshold = 0.000
    feat_df = ct.setTarget(new_data_df, "Long", beLongThreshold)
    
    print(feat_df.tail(2))

    # Adding features with new day
    input_dict = sysUtil.get_dict(system_directory, 'input_dict')

    feat_df = featureGen.generate_features(feat_df, input_dict)
    feat_df = transf.normalizer(feat_df, 'Volume', 50)
    
    col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
    to_drop = ['Open','High','Low', 'gainAhead', 'Close', 'Volume', 'AdjClose', 'beLong']
    for x in to_drop:
        col_vals.append(x)
    model_data = dSet.drop_columns(feat_df, col_vals)
    
    # Retrieve model
    best_model_name = "SVM"
    best_model_segment = "segment-0"
    #best_model_name = system_dict["best_model"]
    file_title = "fit-model-" + best_model_name + "-IS-" + system_name + "-" + best_model_segment +".sav"
    file_name = os.path.join(system_directory, file_title)
    model = pickle.load(open(file_name, 'rb'))
    
    # get last row of data
    lastRow = model_data.shape[0]
    model_end_date = model_data.index[lastRow-1]
    print(model_end_date)
    
    # Make prediction
    predict_data = model_data.iloc[lastRow-1]
    dX = np.zeros_like(predict_data)
    dX = predict_data.values
    dX = dX.reshape(1, -1)
    
    # get prediction
    y_validate = []
    y_validate = model.predict(dX)
    print(y_validate[0])
    
    shadow_trades = dSet.read_csv(system_directory,
                                  system_name,
                                  'OOS_Equity',
                                  'dbd'
                                  )
    
    # Update shadow trades
    new_st_df = shadow_trades.append({'Date'      : tradeDate,
                                      'signal'    : y_validate[0],
                                      'gainAhead' : 0.000,
                                      'Close'     : feat_df.Close[lastRow-1]
                                      },
                                      ignore_index=True
                                      )
    
    new_st_df['gainAhead'] = ct.gainAhead(new_st_df.Close)
    
    # save updated shadow trades
    dSet.save_csv(system_directory,
                  system_name,
                  'OOS_Equity',
                  'dbd', 
                  new_st_df
                  )

    # Load TMS Part 1
    tms1 = dSet.read_csv(system_directory,
                         system_name,
                         'TMS_Part1',
                         'dbd'
                         )
    
    # Update TMS-Part 1 data with latest date
    sst = tms1.append({'Date' : tradeDate , 'signal' : y_validate[0], 'gainAhead' : 0, 'Close' :  feat_df.Close[lastRow-1]} , ignore_index=True)
    sst.tail(3)
    
    # Update with gainAhead
    end = sst.index[-2]
    sst.iloc[end,sst.columns.get_loc('gainAhead')] = new_st_df.iloc[end,new_st_df.columns.get_loc('gainAhead')]
    sst.tail(3)
    
    # Update TMS Part1 with safe-f, CAR25
    nrows = sst.shape[0]
    # sst = sst.set_index(pd.DatetimeIndex(sst['Date']))
    start = sst.index[0]
    end = sst.index[-1]
    
    iStart = sst.index.get_loc(end)-1
    iEnd = sst.index.get_loc(end)
    
    # retrieve tms_dict
    tms_dict = sysUtil.get_dict(system_directory, 'tms_dict')
    
    sst = tradeRisk.get_safef_car25(sst, iStart, iEnd, tms_dict)
    
    dSet.save_csv(system_directory,
                  system_name,
                  'TMS_Part1',
                  'dbd',
                  sst
                  )
    
    # Now go to TMS Part 2
    tms2 = dSet.read_csv(system_directory,
                         system_name,
                         'TMS_Part2',
                         'dbd'
                         )
    
    # Append last day form TMS Part 1 to TMS Part 2
    tms21 = tms2.copy()
    sst1 = sst.copy()
    #sst1.reset_index(level=sst1.index.names, inplace=True)
    tms21.loc[sst1.index[-1]] = sst1.iloc[-1]
    
    end = tms21.index[-1]
    
    iStart = tms21.index.get_loc(end)-1
    iEnd = tms21.index.get_loc(end)
    
    # Update gainAhead
    tms21.iloc[iStart,tms21.columns.get_loc('gainAhead')] = sst1.iloc[iStart,sst1.columns.get_loc('gainAhead')]
    
    tms21 = tradeRisk.update_tms_trade_dec(tms21, iStart, iEnd, y_validate)
        
    print(tms21.tail(4))
        
    dSet.save_csv(system_directory,
                  system_name,
                  'TMS_Part2',
                  'dbd',
                  tms21
                  )
    
    system_dict['pivotDate']=tradeDate
    sysUtil.save_dict(system_name, 'system_dict', system_dict)
    
    plotIt.plot_equity_drawdown(issue, tms21)
    
    plotIt.plot_CAR25_close(issue, tms21)
    
    plotTitle = "Safe-f for " + issue
    plotIt.plot_v1(tms21['safef'][:-2], plotTitle)