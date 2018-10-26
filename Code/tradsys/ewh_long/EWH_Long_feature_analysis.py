# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

EWH_Long_feature_analysis.py
"""
import sys
sys.path.append('../../lib')
sys.path.append('../../utilities')

from plot_utils import *
from time_utils import *
from retrieve_data import *
from transformers import *
from model_utils import *
#from stat_tests import *

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os.path
import pickle
from sklearn.externals import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BDay

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    ct = ComputeTarget()
    modelUtil = ModelUtility()
    
    # Load issue data    
    issue = "EWH"
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue) 
    
    #set beLong level
    beLongThreshold = 0.000
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    
    features_to_drop = []
    features_to_drop.extend(('Open','High','Low','Close', 'gainAhead', 'Symbol'))

    feature_dict = {}
    ######################
    # Transform data
    ######################
    predictor_vars = 'Price and percReturn lags'
     # set lag on Close (Close)
    transf = Transformers()
    lag_var = 'Close'
    lags = 4
    dataSet = transf.add_lag(dataSet, lag_var, lags)
    
    # set % return variables and lags
    dataSet["percReturn"] = dataSet["Close"].pct_change()*100
    lag_var = 'percReturn'
    lags = 5    
    dataSet = transf.add_lag(dataSet, lag_var, lags) 
    features_to_drop.append(lag_var)
    
    # add Close Higher features
    dataSet['1DayHigherClose'] = dataSet['Close'] > dataSet['Close_lag1']
    dataSet['2DayHigherClose'] = dataSet['Close'] > dataSet['Close_lag2']
    dataSet['3DayHigherClose'] = dataSet['Close'] > dataSet['Close_lag3']
    dataSet['4DayHigherClose'] = dataSet['Close'] > dataSet['Close_lag4']
    
    dataSet['1DayLowerClose'] = dataSet['Close'] < dataSet['Close_lag1']
    dataSet['2DayLowerClose'] = dataSet['Close'] < dataSet['Close_lag2']
    dataSet['3DayLowerClose'] = dataSet['Close'] < dataSet['Close_lag3']
    dataSet['4DayLowerClose'] = dataSet['Close'] < dataSet['Close_lag4']
    
    #############################
    # Set IS-OOS parameters
    ##############################
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 3
    oos_months = 6
    segments = 3
    
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    
    is_end_date = is_start_date + relativedelta(months=is_months)
    oos_end_date = oos_start_date + relativedelta(months=oos_months)
    
    modelStartDate = is_start_date
    modelEndDate = modelStartDate + relativedelta(months=is_months)
    
    #dataSet = read_issue_data(issue, dataLoadStartDate, pivotDate)
    print(issue)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    
    # save Dataset
    print("====Saving dataSet====")
    modelname = 'RF'
    file_title = issue + "_in_sample_" + modelname + ".pkl"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    dataSet.to_pickle(file_name)
    
    #dataSet = dSet.drop_columns(dataSet,features_to_drop)
    
    # Set data set for analysls
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,pivotDate)

    model_results = []
    
    # IS only
    for i in range(segments):
        mmData = dataSet[modelStartDate:modelEndDate]
        
        plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
        plotIt.plot_v2x(mmData, plotTitle)
        plotIt.histogram(mmData['beLong'], x_label="beLong signal", y_label="Frequency", 
          title = "beLong distribution for " + issue)        
        plt.show(block=False)
        
        ######################
        # ML section
        ######################
        #  Make 'iterations' index vectors for the train-test split
        iterations = 50
        info_dict = {'issue':issue, 'modelStartDate':modelStartDate, 'modelEndDate':modelEndDate, 'modelname':modelname, 'nrows':nrows, 'predictors':predictor_vars}     
        
        dX, dy = modelUtil.prepare_for_classification(mmData)        
        
        sss = StratifiedShuffleSplit(n_splits=iterations,test_size=0.33, random_state=None)
                
        model = RandomForestClassifier(n_jobs=-1, random_state=55, min_samples_split=10 , n_estimators=500, max_features = 'auto', min_samples_leaf = 10, oob_score = 'TRUE')
                
        ### add issue, other data OR use dictionary to pass data!!!!!!
        model_results = modelUtil.model_and_test(dX, dy, model, model_results, sss, info_dict)
    
        modelStartDate = modelEndDate  + BDay(1)
        modelEndDate = modelStartDate + relativedelta(months=is_months) - BDay(1)

        
    df = pd.DataFrame(model_results)
    df = df[['Issue','StartDate','EndDate','Model','Rows','beLongCount','Predictors','IS-Accuracy','IS-Precision','IS-Recall','IS-F1','OOS-Accuracy','OOS-Precision','OOS-Recall','OOS-F1']]
    print(df)

    dirext = issue + '_Model_' + modelname + '_start_' + str(dataLoadStartDate) + '_end_' + str(pivotDate) + '_' + datetime.datetime.now().strftime("%Y-%m-%d")
    print(dirext)
    filename = "IS_model_iteration_" + dirext + ".csv"
    current_directory = os.getcwd()
    df.to_csv(current_directory+"\\"+filename, encoding='utf-8', index=False)
    
    
    # save Dataset
    print("====Saving model====")
    
    modelname = 'RF'
    file_title = issue + "_predictive_model_" + modelname + ".sav"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    #joblib.dump(model,filename)
    pickle.dump(model, open(file_name, 'wb'))
    
        
        
        
        
        