# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

model_utils.py
"""
from retrieve_data import *

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class ModelUtility:
    
    def conf_matrix_results(self, cm_results):
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
    
    def prepare_for_classification(self, mmData):
        datay = mmData['beLong']
        dataX = mmData.drop(['beLong'],axis=1)
        #  Copy from pandas dataframe to numpy arrays
        dy = np.zeros_like(datay)
        dX = np.zeros_like(dataX)
        dy = datay.values
        dX = dataX.values
        return dX, dy
    
    def model_and_test(self, dX, dy, model, model_results, sss, info_dict):
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

        
        print ("\n\nSymbol is ", info_dict['issue'])
        print ("Learning algorithm is ", info_dict['modelname'])
        #print ("Confusion matrix for %i randomized tests" % iterations)
        print ("for dates ", info_dict['modelStartDate'], " through ", info_dict['modelEndDate']) 
        print ("----------------------")
        print ("==In Sample==")
        cm_results_is = self.conf_matrix_results(cm_sum_is)
        print ("\n----------------------")
        print ("\n==Out Of Sample==")
        cm_results_oos = self.conf_matrix_results(cm_sum_oos)
        print ("\n----------------------")
        
        
        model_results.append({'Issue': info_dict['issue'], 'StartDate': info_dict['modelStartDate'].strftime("%Y-%m-%d"), 'EndDate': info_dict['modelEndDate'].strftime("%Y-%m-%d"), 'Model': info_dict['modelname'], 'Rows': info_dict['nrows'], 'beLongCount': str(np.count_nonzero(dy==1)), 'Predictors': info_dict['predictors'], 'IS-Accuracy': np.mean(accuracy_scores_is), 'IS-Precision': np.mean(precision_scores_is), 'IS-Recall': np.mean(recall_scores_is), 'IS-F1': np.mean(f1_scores_is), 'OOS-Accuracy':  np.mean(accuracy_scores_oos), 'OOS-Precision': np.mean(precision_scores_oos), 'OOS-Recall': np.mean(recall_scores_oos), 'OOS-F1': np.mean(f1_scores_oos)})
        
        return model_results

if __name__ == "__main__":
    from plot_utils import *
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    plotIt = PlotUtility()
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(dataSet['Close'], plotTitle)
    
    beLongThreshold = 0
    cT = ComputeTarget()
    mmData = cT.setTarget(dataSet, "Long", beLongThreshold)
    
    addIndic1 = Indicators()
    ind_list = [("RSI", 2.3),("ROC",5),("DPO",5),("ATR", 5)]
    dataSet = addIndic1.add_indicators(dataSet, ind_list)
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    #fig = plt.figure(figsize=(15,8  ))
    fig, axes = plt.subplots(5,1, figsize=(15,8), sharex=True)

    axes[0].plot(rsiDataSet['Close'], label=issue)
    axes[1].plot(rsiDataSet['Close_RSI'], label='RSI');
    axes[2].plot(rsiDataSet['Close_ROC'], label='ROC');
    axes[3].plot(rsiDataSet['Close_DPO'], label='DPO');
    axes[4].plot(rsiDataSet['Close_ATR'], label='ATR');
    
    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0)
    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axes:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=False)
    
    
