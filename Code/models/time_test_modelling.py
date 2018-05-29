# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

time_test_modelling.py
"""
import sys
sys.path.append('../lib')
sys.path.append('../transform')
sys.path.append('../indicators')
sys.path.append('../predictors')
sys.path.append('../utilities')

# Import the Time Series library
import statsmodels.tsa.stattools as ts
from retrieve_issue_data import read_issue_data
from stat_tests import *
from compute_target import *
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
plt.style.use('seaborn-ticks')


if __name__ == "__main__":
    issue = "tlt"
    pivotDate = datetime.date(2018, 4, 1)
    inSampleOutOfSampleRatio = 2
    outOfSampleMonths = 2
    inSampleMonths = inSampleOutOfSampleRatio * outOfSampleMonths
    segments = 2
    months_to_load = inSampleMonths + segments * outOfSampleMonths
       
    inSampleStartDate = pivotDate - relativedelta(months=months_to_load)
    dataLoadStartDate = inSampleStartDate - relativedelta(months=1)
    print("Load Date: ", dataLoadStartDate)
    print("In Sample Start  Date: ", inSampleStartDate)
    print("Pivot Date: ", pivotDate)
    
    dataSet = read_issue_data(issue, dataLoadStartDate, pivotDate)
    print(issue)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    
    # set lag on Close (Pri)
    lag_var = 'Pri'
    lags = 5
    dataSet = add_lag(dataSet, lag_var, lags)
    
    # set % return variables and lags
    dataSet["percReturn"] = dataSet["Pri"].pct_change()*100
    lag_var = 'percReturn'
    lags = 5    
    dataSet = add_lag(dataSet, lag_var, lags)    
    
    # Put indicators and transforms here
    
    #set beLong level
    beLongThreshold = 0
    
    modelStartDate = inSampleStartDate
    print(modelStartDate)
    modelEndDate = modelStartDate + relativedelta(months=inSampleMonths)
#    print(modelEndDate)
#    modelData = dataSet.ix[modelStartDate:modelEndDate]
#    print(modelData)

    # IS only
    for i in range(segments):
#        inSampleDataSetName = 'isDataSet' + str(i)
#        print (inSampleDataSetName)
#        outOfSampleDataSetName = 'oosDataSet' + str(i)
#        print (outOfSampleDataSetName)
        modelEndDate = modelStartDate + relativedelta(months=inSampleMonths)
        modelData = dataSet.ix[modelStartDate:modelEndDate].copy()
        #print(modelData)
        
        # set target var
        mmData = setTarget(modelData, "Long", beLongThreshold)
        nrows = mmData.shape[0]
        print ("nrows: ", nrows)
        print (mmData.shape)
        print (mmData.tail(10))
        print ("beLong counts: ")
        print (mmData['beLong'].value_counts())
        print ("out of ", nrows)
        
        mmData = mmData.drop(['Open','High','Low','Close'],axis=1)
        
        plt.style.use('seaborn-ticks')
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.plot(mmData['Pri'], label=issue)
        plt.legend(loc='upper left')
        print("\n\n\n")
        plt.show(block=False)
        
        mmData = mmData.drop(['Pri'],axis=1)
        
        datay = mmData['beLong']
        nrows = datay.shape[0]
        print ("nrows beLong: ", nrows)
        print(datay.head())
        plt.figure(2)
        datay.hist(figsize=(8,4  ))
        print("\n\n\n")
        plt.show(block=False)
        
        mmData = mmData.drop(['beLong'],axis=1)
        dataX = mmData
        
        #  Copy from pandas dataframe to numpy arrays
        dy = np.zeros_like(datay)
        dX = np.zeros_like(dataX)
        
        dy = datay.values
        dX = dataX.values
        
        ######################
        # ML section
        
        iterations = 100
        
        model = LogisticRegression()
        
        #  Make 'iterations' index vectors for the train-test split
        sss = StratifiedShuffleSplit(dy,iterations,test_size=0.33, random_state=None)
        
        model_results = []
        
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
            
        #  For each entry in the set of splits, fit and predict
        for train_index,test_index in sss:
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
        
        tpIS = cm_sum_is[1,1]
        fnIS = cm_sum_is[1,0]
        fpIS = cm_sum_is[0,1]
        tnIS = cm_sum_is[0,0]
        precisionIS = tpIS/(tpIS+fpIS)
        recallIS = tpIS/(tpIS+fnIS)
        accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
        f1IS = (2.0 * precisionIS * recallIS) / (precisionIS+recallIS) 
        
        tpOOS = cm_sum_oos[1,1]
        fnOOS = cm_sum_oos[1,0]
        fpOOS = cm_sum_oos[0,1]
        tnOOS = cm_sum_oos[0,0]
        precisionOOS = tpOOS/(tpOOS+fpOOS)
        recallOOS = tpOOS/(tpOOS+fnOOS)
        accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
        f1OOS = (2.0 * precisionOOS * recallOOS) / (precisionOOS+recallOOS) 
        
        print ("\n\nSymbol is ", issue)
        print ("Learning algorithm is ", model)
        print ("Confusion matrix for %i randomized tests" % iterations)
        print ("for dates ", startDate, " through ", endDate) 
        print ("----------------------")
        print ("==In Sample==")
        print ("     Predicted")
        print ("      pos neg")
        print ("pos:  %i  %i  Recall:%.2f" % (tpIS, fnIS, recallIS))
        print ("neg:  %i  %i" % (fpIS, tnIS))
        print ("Prec: %.2f          Accuracy: %.2f " % (precisionIS, accuracyIS))
        print ("f1:   %.2f" % f1IS)
        print ("\n----------------------")
        print ("\n==Out Of Sample==")
        print ("     Predicted")
        print ("      pos neg")
        print ("pos:  %i  %i  Recall:%.2f" % (tpOOS, fnOOS, recallOOS))
        print ("neg:  %i  %i" % (fpOOS, tnOOS))
        print ("Prec: %.2f          Accuracy: %.2f " % (precisionOOS, accuracyOOS))
        print ("f1:   %.2f" % f1OOS)
        print ("\n----------------------")
        
        print ("\n==In Sample==")
        print('Accuracy: %.2f' % np.mean(accuracy_scores_is))
        print('Precision: %.2f' % np.mean(precision_scores_is))
        print('Recall: %.2f' % np.mean(recall_scores_is))
        print('F1: %.2f' % np.mean(f1_scores_is))
        print ("\n==Out Of Sample==")
        print('Accuracy: %.2f' % np.mean(accuracy_scores_oos))
        print('Precision: %.2f' % np.mean(precision_scores_oos))
        print('Recall: %.2f' % np.mean(recall_scores_oos))
        print('F1: %.2f' % np.mean(f1_scores_oos))
        print ("\nend of run")

        
        
        
        
        
        
        
        