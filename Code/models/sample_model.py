# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:25:39 2018

@author: kruegkj

sample_model.py

First pass on modeling of input dataframe

Inputs:
    Dataframe
    Timeframe?
    Model names
    Variables to drop?
Outputs:
    Model results
    Saved model?
    
From main, create many of the vairables created as defaults in other modules

"""
import sys
sys.path.append('../lib')
sys.path.append('../transform')
sys.path.append('../indicators')
sys.path.append('../predictors')
sys.path.append('../utilities')
from predictors_main import *
from transformers_main import *
from plot_utils import *
from retrieve_issue_data import read_issue_data

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.naive_bayes import MultinomialNB
#from sklearn import neighbors
#from sklearn import linear_model
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import datetime
plt.style.use('seaborn-ticks')


if __name__ == "__main__":
    # Get issue data
    issue = "tlt"
    lookback = 16
    dataLoadStartDate = "2014-01-01"
    dataLoadEndDate = "2018-03-30" 
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    print(issue)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plot_v1(dataSet['Close'], plotTitle)
    
    #set beLong level
    beLongThreshold = 0
    # set target var
    dataSet = setTarget(dataSet, "Long", beLongThreshold)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    print (dataSet.shape)
    print (dataSet.tail(10))
    print ("beLong counts: ")
    print (dataSet['beLong'].value_counts())
    print ("out of ", nrows)
    
    # set lag on Close (Pri)
    lag_var = 'Pri'
    lags = 5
    dataSet = add_lag(dataSet, lag_var, lags)
    
    # set % return variables and lags
    dataSet["percReturn"] = dataSet["Pri"].pct_change()*100
    lag_var = 'percReturn'
    lags = 5    
    dataSet = add_lag(dataSet, lag_var, lags)
    
    # add indicators
    RSI_lookback = 2.3
    ROC_lookback = 5
    DPO_lookback = 5
    ATR_lookback = 5
    ind_list = [("RSI", RSI_lookback),("ROC",ROC_lookback),("DPO",DPO_lookback),("ATR", ATR_lookback)]
    dataSet = add_indicators(dataSet, ind_list)
    
    #normalize     
    zScore_lookback = 3
    dataSet = zScore_transform(dataSet, zScore_lookback, 'Pri_ROC')
    dataSet = zScore_transform(dataSet, zScore_lookback, 'Pri_DPO')
    dataSet = zScore_transform(dataSet, zScore_lookback, 'Pri_ATR')
    
    mData = dataSet.drop(['Open','High','Low','Close','gainAhead'],axis=1)
    
    startDate = "2017-03-01"
    endDate = "2017-09-30"
    
    mmData = mData.ix[startDate:endDate]
    
    mmData = mmData.drop(['Pri'],axis=1)
    
    datay = mmData.beLong
    nrows = datay.shape[0]
    print ("nrows beLong: ", nrows)
    print(datay.head())
    datay.hist(figsize=(8,4  ))
    histogram(datay, "", "", "beLong counts")
    
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
    
    model_results.append({'Issue': issue, 'StartDate': startDate, 'EndDate': endDate, 'Model': model, 'RSILookback': RSI_lookback, 'IS-Accuracy': np.mean(accuracy_scores_is), 'IS-Precision': np.mean(precision_scores_is), 'IS-Recall': np.mean(recall_scores_is), 'IS-F1': np.mean(f1_scores_is), 'OOS-Accuracy':  np.mean(accuracy_scores_oos), 'OOS-Precision': np.mean(precision_scores_oos), 'OOS-Recall': np.mean(recall_scores_oos), 'OOS-F1': np.mean(f1_scores_oos)})

df = pd.DataFrame(model_results)
df = df[['Issue','StartDate','EndDate','Model','RSILookback','IS-Accuracy','IS-Precision','IS-Recall','IS-F1','OOS-Accuracy','OOS-Precision','OOS-Recall','OOS-F1']]
print(df)

dirext = issue + '_RSILookback' + str(RSI_lookback) + '_start' + str(startDate) + '_end' + str(endDate) + '_' +\
                datetime.now().strftime("%Y-%m-%d")
filename = "model_iteration_" + dirext + ".csv" 
df.to_csv(filename, encoding='utf-8', index=False)


        
    
    
    
    