# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:32:53 2018

@author: KRUEGKJ

logreg_model_investigation.py
"""

import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import *
from retrieve_data import *
from indicators import *
from transformers import *
#from stat_tests import *

# Import the Time Series library
#import statsmodels.tsa.stattools as ts
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
#plt.style.use('seaborn-ticks')
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BDay
#import matplotlib as mpl
#plt.style.use('seaborn-ticks')
#import matplotlib.ticker as ticker

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import os.path

issue = "TLT"
us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
pivotDate = datetime.date(2018, 2, 2)
inSampleOutOfSampleRatio = 4
outOfSampleMonths = 2
inSampleMonths = inSampleOutOfSampleRatio * outOfSampleMonths
print("inSampleMonths: " + str(inSampleMonths))
months_to_load = outOfSampleMonths + segments * inSampleMonths
print("Months to load: " + str(months_to_load))
   
inSampleStartDate = pivotDate - relativedelta(months=months_to_load)
dataLoadStartDate = inSampleStartDate - relativedelta(months=1)
print("Load Date: ", dataLoadStartDate)
print("In Sample Start  Date: ", inSampleStartDate)
print("Pivot Date: ", pivotDate)

dSet = DataRetrieve()
dataSet = dSet.read_issue_data(issue)   
dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,pivotDate)

#dataSet = read_issue_data(issue, dataLoadStartDate, pivotDate)
print(issue)
nrows = dataSet.shape[0]
print ("nrows: ", nrows)

addIndic1 = Indicators()
ind_list = [("RSI", 2.3),("ROC",5),("DPO",5),("ATR", 5)]
dataSet = addIndic1.add_indicators(dataSet, ind_list)
 
zScore_lookback = 3
transf = Transformers()
transfList = ['Pri_ROC','Pri_DPO','Pri_ATR']
for i in transfList:
    dataSet = transf.zScore_transform(dataSet, zScore_lookback, i)

# set lag on Close (Pri)
transf = Transformers()
lag_var = 'Pri'
lags = 2
dataSet = transf.add_lag(dataSet, lag_var, lags)

lags = 2
transfList = ['Pri_ROC_zScore','Pri_DPO_zScore','Pri_ATR_zScore']
for i in transfList:
    dataSet = transf.add_lag(dataSet, i, lags)


# set % return variables and lags
dataSet["percReturn"] = dataSet["Pri"].pct_change()*100
lag_var = 'percReturn'
lags = 2    
dataSet = transf.add_lag(dataSet, lag_var, lags)    

predictor_vars = 'Price and percReturn lags'
# Put indicators and transforms here

#set beLong level
beLongThreshold = 0.0
ct = ComputeTarget()

modelStartDate = inSampleStartDate
modelEndDate = modelStartDate + relativedelta(months=inSampleMonths)

plotIt = PlotUtility()

df2 = pd.date_range(start=modelStartDate, end=modelEndDate, freq=us_cal)
modelData = dataSet.reindex(df2)

# set target var
mmData = ct.setTarget(modelData, "Long", beLongThreshold)
nrows = mmData.shape[0]
print ("beLong counts: ")
be_long_count = mmData['beLong'].value_counts()
print (be_long_count)
print ("out of ", nrows)

mmData = mmData.drop(['Open','High','Low','Close', 'percReturn', 'gainAhead', 'Symbol','Pri_ROC','Pri_DPO','Pri_ATR'],axis=1)

plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
plotIt.plot_v2x(mmData['Pri'], mmData['beLong'], plotTitle)
plotIt.histogram(mmData['beLong'], x_label="beLong signal", y_label="Frequency", 
  title = "beLong distribution for " + issue)        
plt.show(block=False)

mmData = mmData.drop(['Pri'],axis=1)

# Chnage -1 to 0 for log reg
#mmData.beLong.replace([1, -1], [1, 0], inplace=True)
datay = mmData['beLong']
nrows = datay.shape[0]
print ("nrows beLong: ", nrows)


mmData = mmData.drop(['beLong'],axis=1)
dataX = mmData

# get names of features
names = dataX.columns.values.tolist()

# correlation plot
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
plot_corr(dataX,8)

#  Copy from pandas dataframe to numpy arrays
dy = np.zeros_like(datay)
dX = np.zeros_like(dataX)

dy = datay.values
dX = dataX.values

#########################################
#import statsmodels.api as sm
#logit_model=sm.Logit(datay,dataX)
#result=logit_model.fit()
#print(result.summary())
#####################################

######################
# ML section
model_results = []

iterations = 10

model = RandomForestClassifier(n_jobs=2, random_state=0, min_samples_split=10, n_estimators=1000, max_features = 'sqrt')
modelname = 'RF'

#  Make 'iterations' index vectors for the train-test split
sss = StratifiedShuffleSplit(n_splits=iterations,test_size=0.33, random_state=None)

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
    #for i in range(0, 5):
    #    print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_train)[i], y_pred_is[i]))
    cm_is = confusion_matrix(y_train, y_pred_is)
    cm_sum_is = cm_sum_is + cm_is
    accuracy_scores_is.append(accuracy_score(y_train, y_pred_is))
    precision_scores_is.append(precision_score(y_train, y_pred_is))
    recall_scores_is.append(recall_score(y_train, y_pred_is))
    f1_scores_is.append(f1_score(y_train, y_pred_is))
    
#  test the out-of-sample data
    y_pred_oos = model.predict(X_test)
    #for i in range(0, 5):
    #    print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], y_pred_oos[i]))
    cm_oos = confusion_matrix(y_test, y_pred_oos)
    cm_sum_oos = cm_sum_oos + cm_oos
    accuracy_scores_oos.append(accuracy_score(y_test, y_pred_oos))
    precision_scores_oos.append(precision_score(y_test, y_pred_oos))
    recall_scores_oos.append(recall_score(y_test, y_pred_oos))
    f1_scores_oos.append(f1_score(y_test, y_pred_oos))

print (sorted(zip(map(lambda x: round(x, 2), model.feature_importances_), names), reverse=True))

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
print ("for dates ", modelStartDate, " through ", modelEndDate) 
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

model_results.append({'Issue': issue, 'StartDate': modelStartDate.strftime("%Y-%m-%d"), 'EndDate': modelEndDate.strftime("%Y-%m-%d"), 'Model': modelname, 'Rows': nrows, 'beLongCount': str(np.count_nonzero(dy==1)), 'Predictors': predictor_vars, 'IS-Accuracy': np.mean(accuracy_scores_is), 'IS-Precision': np.mean(precision_scores_is), 'IS-Recall': np.mean(recall_scores_is), 'IS-F1': np.mean(f1_scores_is), 'OOS-Accuracy':  np.mean(accuracy_scores_oos), 'OOS-Precision': np.mean(precision_scores_oos), 'OOS-Recall': np.mean(recall_scores_oos), 'OOS-F1': np.mean(f1_scores_oos)})
        
df = pd.DataFrame(model_results)
df = df[['Issue','StartDate','EndDate','Model','Rows','beLongCount','Predictors','IS-Accuracy','IS-Precision','IS-Recall','IS-F1','OOS-Accuracy','OOS-Precision','OOS-Recall','OOS-F1']]
print(df)

dirext = issue + '_Model_' + modelname + '_start_' + str(dataLoadStartDate) + '_end_' + str(pivotDate) + '_TS_' + datetime.datetime.now().strftime("%Y-%m-%d")
print(dirext)
filename = "IS_model_iteration_" + dirext + ".csv"
current_directory = os.getcwd()
df.to_csv(current_directory+"\\"+filename, encoding='utf-8', index=False)
    
        
        
        
        
        