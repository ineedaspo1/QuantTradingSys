# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

insample_sample_model.py
"""
import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import *
from time_utils import *
from retrieve_data import *
from candle_indicators import *
from transformers import *
from ta_momentum_studies import *
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

def conf_matrix_results(cm_results):
    return_cm = ()
    tp = cm_results[1,1]
    fn = cm_results[1,0]
    fp = cm_results[0,1]
    tn = cm_results[0,0]
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tp + fn + fp + tn)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return_cm = (precision, recall, accuracy, f1)
    
    print ("     Predicted")
    print ("      pos neg")
    print ("pos:  %i  %i  Recall:%.2f" % (tp, fn, recall))
    print ("neg:  %i  %i" % (fp, tn))
    print ("Prec: %.2f          Accuracy: %.2f " % (precision, accuracy))
    print ("f1:   %.2f" % f1)
    return return_cm

if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    ct = ComputeTarget()
    candle_ind = CandleIndicators()
    dSet = DataRetrieve()
    taLibMomSt = TALibMomentumStudies()
    transf = Transformers()
    
    feature_dict = {}
    
    # Load issue data
    issue = "TLT"
    dataSet = dSet.read_issue_data(issue) 
    
    #set beLong level
    beLongThreshold = 0.000
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)

    ######################
    # Transform data
    ######################
    predictor_vars = 'Price and percReturn lags'
     # set lag on Close (Pri)
#    transf = Transformers()
#    lag_var = 'Close'
#    lags = 4
#    dataSet = transf.add_lag(dataSet, lag_var, lags)
    
    # set % return variables and lags
    dataSet["percReturn"] = dataSet["Close"].pct_change()*100
    lag_var = 'percReturn'
    lags = 5    
    dataSet = transf.add_lag(dataSet, lag_var, lags) 
    
#    days_to_plot = 4
#    for i in range(1, days_to_plot + 1):
#        num_days = i
#        dataSet = candle_ind.higher_close(dataSet, num_days)
#        dataSet = candle_ind.lower_close(dataSet, num_days)
    #dataSet = taLibMomSt.RSI(dataSet, 20)
    dataSet = taLibMomSt.PPO(dataSet, 10, 24)
#    dataSet = taLibMomSt.CMO(dataSet, 20)
    dataSet = taLibMomSt.CCI(dataSet, 20)
    
    # Set IS-OOS parameters
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 4
    oos_months = 3
    segments = 3
    
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    
    is_end_date = is_start_date + relativedelta(months=is_months)
    oos_end_date = oos_start_date + relativedelta(months=oos_months)
    
    modelStartDate = is_start_date
    print(modelStartDate)
    modelEndDate = modelStartDate + relativedelta(months=is_months)
    print(modelEndDate)
    #dataSet = read_issue_data(issue, dataLoadStartDate, pivotDate)
    print(issue)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    
    # save Dataset of analysis
    print("====Saving dataSet====")
    modelname = 'RF'
    file_title = issue + "_in_sample_" + modelname + ".pkl"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    dataSet.to_pickle(file_name)
    
    # Set data set for analysls
    dataSet2 = dSet.set_date_range(dataSet, dataLoadStartDate, pivotDate)

    model_results = []
    
    # IS only
    for i in range(segments):
        accuracy_scores_is = []
        accuracy_scores_oos = []
        precision_scores_is = []
        precision_scores_oos = []
        recall_scores_is = []
        recall_scores_oos = []
        f1_scores_is = []
        f1_scores_oos = []
        
        #  Initialize the confusion matrix
        cm_sum_is = np.zeros((2,2))
        cm_sum_oos = np.zeros((2,2))
        
        mmData = dataSet2[modelStartDate:modelEndDate]
        
        # set target var      
        nrows = mmData.shape[0]
        #print ("nrows: ", nrows)
        #print (mmData.shape)
        #print (mmData.tail(10))
        print ("beLong counts: ")
        be_long_count = mmData['beLong'].value_counts()
        print (be_long_count)
        print ("out of ", nrows)
        
        mmData = mmData.drop(['Open','High','Low', 'gainAhead', 'Symbol', 'percReturn'],axis=1)
        
        plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
        plotIt.plot_v2x(mmData, plotTitle)
        plotIt.histogram(mmData['beLong'], x_label="beLong signal", y_label="Frequency", title = "beLong distribution for " + issue)        
        plt.show(block=False)
        
        mmData = mmData.drop(['Close','Date'],axis=1)
        
        datay = mmData['beLong']
        nrows = datay.shape[0]
        print ("nrows beLong: ", nrows)
        
        mmData = mmData.drop(['beLong'],axis=1)
        dataX = mmData
        
        #############
        # Correlation examination
        #####################
        # get names of features
        names = dataX.columns.values.tolist()
        # correlation plot        
        def correlation_matrix(df,size=10):
            from matplotlib import pyplot as plt
            from matplotlib import cm as cm
            fig = plt.figure(figsize=(size, size))
            ax1 = fig.add_subplot(111)
            cmap = cm.get_cmap('jet', 30)
            corr = df.corr()
            cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
            ax1.grid(True)
            plt.title('Feature Correlation')
            plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
            plt.yticks(range(len(corr.columns)), corr.columns);
            # Add colorbar, make sure to specify tick locations to match desired ticklabels
            fig.colorbar(cax, ticks=[-1, -.5, 0, .5 ,1])
            plt.show()
        correlation_matrix(dataX)
        
        #  Copy from pandas dataframe to numpy arrays
        dy = np.zeros_like(datay)
        dX = np.zeros_like(dataX)
        
        dy = datay.values
        dX = dataX.values
        
        ######################
        # ML section
        iterations = 5
#        from sklearn.svm import SVC 
#        model = SVC(kernel='linear')
        model = RandomForestClassifier(n_jobs=-1, random_state=55, min_samples_split=10 , n_estimators=500, max_features = 'auto', min_samples_leaf = 10, oob_score = 'TRUE')
        
        #  Make 'iterations' index vectors for the train-test split
        sss = StratifiedShuffleSplit(n_splits=iterations,test_size=0.33, random_state=None)
        
        #  For each entry in the set of splits, fit and predict
        for train_index,test_index in sss.split(dX,dy):
            X_train, X_test = dX[train_index], dX[test_index]
            y_train, y_test = dy[train_index], dy[test_index] 
            
            #  fit the model to the in-sample data
            model.fit(X_train, y_train)
            
            #  test the in-sample fit    
            y_pred_is = model.predict(X_train)
            cm_is = confusion_matrix(y_train, y_pred_is)
            cm_sum_is = cm_sum_is + cm_is
            accuracy_scores_is.append(accuracy_score(y_train, y_pred_is))
            precision_scores_is.append(precision_score(y_train, y_pred_is))
            recall_scores_is.append(recall_score(y_train, y_pred_is))
            f1_scores_is.append(f1_score(y_train, y_pred_is))
            
            #  test the out-of-sample data
            y_pred_oos = model.predict(X_test)
            cm_oos = confusion_matrix(y_test, y_pred_oos)
            cm_sum_oos = cm_sum_oos + cm_oos
            accuracy_scores_oos.append(accuracy_score(y_test, y_pred_oos))
            precision_scores_oos.append(precision_score(y_test, y_pred_oos))
            recall_scores_oos.append(recall_score(y_test, y_pred_oos))
            f1_scores_oos.append(f1_score(y_test, y_pred_oos))
     
        
        print (sorted(zip(map(lambda x: round(x, 2), model.feature_importances_), names), reverse=True))
        
        
        print ("\n\nSymbol is ", issue)
        print ("Learning algorithm is ", model)
        print ("Confusion matrix for %i randomized tests" % iterations)
        print ("for dates ", modelStartDate, " through ", modelEndDate) 
        print ("----------------------")
        print ("==In Sample==")
        cm_results_is = conf_matrix_results(cm_sum_is)
        print ("\n----------------------")
        print ("\n==Out Of Sample==")
        cm_results_oos = conf_matrix_results(cm_sum_oos)
        print ("\n----------------------")
        
        
        model_results.append({'Issue': issue, 'StartDate': modelStartDate.strftime("%Y-%m-%d"), 'EndDate': modelEndDate.strftime("%Y-%m-%d"), 'Model': modelname, 'Rows': nrows, 'beLongCount': str(np.count_nonzero(dy==1)), 'Predictors': predictor_vars, 'IS-Accuracy': np.mean(accuracy_scores_is), 'IS-Precision': np.mean(precision_scores_is), 'IS-Recall': np.mean(recall_scores_is), 'IS-F1': np.mean(f1_scores_is), 'OOS-Accuracy':  np.mean(accuracy_scores_oos), 'OOS-Precision': np.mean(precision_scores_oos), 'OOS-Recall': np.mean(recall_scores_oos), 'OOS-F1': np.mean(f1_scores_oos)})
        
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
