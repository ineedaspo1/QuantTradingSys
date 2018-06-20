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

issue = "XLY"

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
pivotDate = datetime.date(2018, 2, 2)
inSampleOutOfSampleRatio = 2
outOfSampleMonths = 8
inSampleMonths = inSampleOutOfSampleRatio * outOfSampleMonths
print("inSampleMonths: " + str(inSampleMonths))
segments = 1
months_to_load = outOfSampleMonths + segments * inSampleMonths
print("Months to load: " + str(months_to_load))
   
inSampleStartDate = pivotDate - relativedelta(months=months_to_load)
dataLoadStartDate = inSampleStartDate - relativedelta(months=1)
ooSampleStartDate = pivotDate - relativedelta(months=outOfSampleMonths)
print("Load Date: ", dataLoadStartDate)
print("In Sample Start  Date: ", inSampleStartDate)
print("Out Of Sample Start  Date: ", ooSampleStartDate)
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
 
zScore_lookback = 5
transf = Transformers()
transfList = ['Pri_ROC','Pri_DPO','Pri_ATR']
for i in transfList:
    dataSet = transf.zScore_transform(dataSet, zScore_lookback, i)
    
lags = 4
transfList = ['Pri_ROC_zScore','Pri_DPO_zScore','Pri_ATR_zScore']
for i in transfList:
    dataSet = transf.add_lag(dataSet, i, lags)

# set lag on Close (Pri)
transf = Transformers()
lag_var = 'Pri'
lags = 4
dataSet = transf.add_lag(dataSet, lag_var, lags)

# add Close Higher features
dataSet['1DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag1']
dataSet['2DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag2']
dataSet['3DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag3']
dataSet['4DayHigherClose'] = dataSet['Pri'] > dataSet['Pri_lag4']

# set % return variables and lags
dataSet["percReturn"] = dataSet["Pri"].pct_change()*100
lag_var = 'percReturn'
lags = 2    
dataSet = transf.add_lag(dataSet, lag_var, lags)    

predictor_vars = 'Price and percReturn lags'
# Put indicators and transforms here

#set beLong level
beLongThreshold = 0.0002
ct = ComputeTarget()
dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)

modelStartDate = inSampleStartDate
modelEndDate = modelStartDate + relativedelta(months=inSampleMonths)

plotIt = PlotUtility()

df2 = pd.date_range(start=modelStartDate, end=modelEndDate, freq=us_cal)
mmData = dataSet.reindex(df2)

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

"""
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
"""

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

model = RandomForestClassifier(n_jobs=-1, random_state=0, min_samples_split=10, n_estimators=500, max_features = 'auto', min_samples_leaf = 10, oob_score = 'TRUE')
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

"""
Adding validation code
"""
# Select the date range
modelStartDate = ooSampleStartDate
modelEndDate = modelStartDate + relativedelta(months=outOfSampleMonths)

df2 = pd.date_range(start=modelStartDate, end=modelEndDate, freq=us_cal)
valData = dataSet.reindex(df2)
tradesData = valData
    
nrows = valData.shape[0]
print ("beLong counts: ")
be_long_count = valData['beLong'].value_counts()
print (be_long_count)
print ("out of ", nrows)

valData = valData.drop(['Open','High','Low','Close', 'percReturn', 'Symbol','Pri_ROC','Pri_DPO','Pri_ATR'],axis=1)

plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
plotIt.plot_v2x(valData['Pri'], valData['beLong'], plotTitle)
plotIt.histogram(valData['beLong'], x_label="beLong signal", y_label="Frequency", 
  title = "beLong distribution for " + issue)        
plt.show(block=False)

valModelData = valData.drop(['Pri','beLong','gainAhead'],axis=1)

valRows = valModelData.shape[0]
print("There are %i data points" % valRows)

# test the validation data
y_validate = []
y_validate = model.predict(valModelData)

# Create best estimate of trades
bestEstimate = np.zeros(valRows)

# You may need to adjust for the first and / or final entry 
for i in range(valRows -1):
    print(valData.gainAhead.iloc[i], y_validate[i])
    if y_validate[i] > 0.0: 
        bestEstimate[i] = valData.gainAhead.iloc[i]
    else:
        bestEstimate[i] = 0.0 
        
# Create and plot equity curve
equity = np.zeros(valRows)
equity[0] = 1.0
for i in range(1,valRows):
    equity[i] = (1+bestEstimate[i])*equity[i-1]
    
print("\nTerminal Weatlh: ", equity[valRows-1])
plt.plot(equity)

print("\n End of Run")

valData['valBeLong'] = pd.Series(y_validate, index=valData.index)


#==========================
import matplotlib as mpl
#from matplotlib import style
plt.style.use('seaborn-ticks')
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(hspace=0.05)
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((6,1), (4,0), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)

ax2.plot(valData['valBeLong'], color='green', alpha =0.6)
ax1.plot(valData['Pri'])
ax3.plot(valData['beLong'], color='purple', alpha =0.6)

ax1.label_outer()
ax2.label_outer()
ax2.tick_params(axis='x',which='major',bottom='on')
ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax2.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax3.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
ax2.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
ax3.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax1.label_outer()
ax1.legend(loc='upper left', frameon=True, fontsize=8)
ax2.label_outer()
ax2.legend(loc='upper left', frameon=True, fontsize=8)
ax3.label_outer()
ax3.legend(loc='upper left', frameon=True, fontsize=8)

#==========================
tradesData['valBeLong'] = pd.Series(y_validate, index=tradesData.index)
tradesData['gain'] = tradesData['Close'] - tradesData['Open']

#  Count the number of rows in the file
nrows = tradesData.shape[0]
print ('There are %0.f rows of data' % nrows)

#  Compute cumulative equity for all days
equityAllSignals = np.zeros(nrows)
equityAllSignals[0] = 1
for i in range(1,nrows):
    equityAllSignals[i] = (1+tradesData.gainAhead[i])*equityAllSignals[i-1]

print ('TWR for all signals is %0.3f' % equityAllSignals[nrows-1])
# add to valData
valData['equityAllSignals'] = pd.Series(equityAllSignals, index=valData.index)
    
#  Compute cumulative equity for days with beLong signals    
equityBeLongSignals = np.zeros(nrows)
equityBeLongSignals[0] = 1
for i in range(1,nrows):
    if (tradesData.beLong[i] > 0):
        equityBeLongSignals[i] = (1+tradesData.gainAhead[i])*equityBeLongSignals[i-1]
    else:
        equityBeLongSignals[i] = equityBeLongSignals[i-1]
valData['equityBeLongSignals'] = pd.Series(equityBeLongSignals, index=valData.index)

#  Compute cumulative equity for days with Validation beLong signals    
equityValBeLongSignals = np.zeros(nrows)
equityValBeLongSignals[0] = 1
for i in range(1,nrows):
    if (tradesData.valBeLong[i] > 0):
        equityValBeLongSignals[i] = (1+tradesData.gainAhead[i])*equityValBeLongSignals[i-1]
    else:
        equityValBeLongSignals[i] = equityValBeLongSignals[i-1]
               
print ('TWR for all days with beLong signals is %0.3f' % equityBeLongSignals[nrows-1])
valData['equityValBeLongSignals'] = pd.Series(equityValBeLongSignals, index=valData.index)

#  Plot the two equity streams
fig = plt.figure(figsize=(12,8))
fig.suptitle(issue + ' Portfolio value in Validation')
ax1 = fig.add_subplot(111)
ax1.plot(valData.equityBeLongSignals, color='green',label='BeLong')
ax1.plot(valData.equityAllSignals, color='blue',label='Equity')
ax1.plot(valData.equityValBeLongSignals, color='purple',label='ValBeLong')

ax1.legend(loc='upper left', frameon=True, fontsize=8)
ax1.label_outer()
ax1.tick_params(axis='x',which='major',bottom='on')
ax1.minorticks_on()
ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(valData.Pri, color='black',alpha=0.6,label='CLOSE',linestyle='--')
ax2.legend(loc='center left', frameon=True, fontsize=8)
ax2.label_outer()
plt.show()

#===================================        
# Getting sample equity curve




#  Evaluation of signals

print ("Starting single run")

#  Status variables

ndays = tradesData.shape[0]

#  Local variables for trading system

initialEquity = 100000
fixedTradeDollars = 10000
commission = 0.005      #  Dollars per share per trade

#  These are scalar and apply to the current conditions

entryPrice = 0
exitPrice = 0

#  These have an element for each day loaded
#  Some will be unnecessary

accountBalance = np.zeros(ndays)
cash = np.zeros(ndays)
sharesHeld = np.zeros(ndays)
tradeGain = []
tradeGainDollars = []
openTradeEquity = np.zeros(ndays)
tradeWinsValue = np.zeros(ndays)
tradeLossesValue = np.zeros(ndays)


iTradeDay = 0
iTradeNumber = 0

#  Day 0 contains the initial values
accountBalance[0] = initialEquity
cash[0] = accountBalance[0]
sharesHeld[0] = 0

#  Loop over all the days loaded
for i in range (1,ndays):
    #  Extract the date
    dt = tradesData.index[i]
    #  Check the date
    datesPass = dt.date()>=modelStartDate and dt.date()<=modelEndDate
    if datesPass:
        iTradeDay = iTradeDay + 1
        if sharesHeld[iTradeDay-1] > 0:
            #  In a long position
            if tradesData.valBeLong[i]<0:
                #  target is -1 -- beFlat 
                #  Exit -- close the trade
                exitPrice = tradesData.Pri[i]
                grossProceeds = sharesHeld[iTradeDay-1] * exitPrice
                commissionAmount = sharesHeld[iTradeDay-1] * commission
                netProceeds = grossProceeds - commissionAmount
                #print("netProceeds: ", netProceeds)
                cash[iTradeDay] = cash[iTradeDay-1] + netProceeds
                accountBalance[iTradeDay] = cash[iTradeDay]
                sharesHeld[iTradeDay] = 0
                iTradeNumber = iTradeNumber+1
                #tradeGain[iTradeNumber] = (exitPrice / (1.0 * entryPrice))    
                tradeGain.append(exitPrice / (1.0 * entryPrice))
                tradeGainDollars.append(((exitPrice / (1.0 * entryPrice))*fixedTradeDollars)-fixedTradeDollars)
                
                pass
            else:
                #  target is +1 -- beLong
                #  Continue long
                sharesHeld[iTradeDay] = sharesHeld[iTradeDay-1]
                cash[iTradeDay] = cash[iTradeDay-1]
                MTMPrice = tradesData.Pri[i]
                openTradeEquity = sharesHeld[iTradeDay] * MTMPrice
                accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                pass
        else:
            #  Currently flat
            if tradesData.valBeLong[i]>0:
                #  target is +1 -- beLong
                #  Enter a new position
                entryPrice = tradesData.Pri[i]
                sharesHeld[iTradeDay] = int(fixedTradeDollars/(entryPrice+commission))
                shareCost = sharesHeld[iTradeDay]*(entryPrice+commission)
                cash[iTradeDay] = cash[iTradeDay-1] - shareCost
                openTradeEquity = sharesHeld[iTradeDay]*entryPrice
                accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                pass
            else:
                #  target is -1 -- beFlat
                #  Continue flat
                cash[iTradeDay] = cash[iTradeDay-1]
                accountBalance[iTradeDay] = cash[iTradeDay] 
                pass
            
#  Format and print results        
 
finalAccountBalance = accountBalance[iTradeDay]
print ('Final account balance: %.2f' %  finalAccountBalance)
numberTradeDays = iTradeDay        
numberTrades = iTradeNumber
print ("Number of trades:", numberTrades)

from pandas import Series

Sequity = Series(accountBalance[0:numberTradeDays-1])

Sequity.plot()

tradeWins = sum(1 for x in tradeGain if float(x) >= 1.0)
tradeLosses = sum(1 for x in tradeGain if float(x) < 1.0 and float(x) > 0)
print("Wins: ", tradeWins)
print("Losses: ", tradeLosses)
print("W/L: ", tradeWins/numberTrades)

tradeWinsValue = sum((x*fixedTradeDollars)-fixedTradeDollars for x in tradeGain if float(x) >= 1.0)
tradeLossesValue = sum((x*fixedTradeDollars)-fixedTradeDollars for x in tradeGain if float(x) < 1.0 and float(x) > 0)
print('Total value of Wins:  %.2f' % tradeWinsValue)
print('Total value of Losses:  %.2f' % tradeLossesValue)
#(Win % x Average Win Size) – (Loss % x Average Loss Size)
print('Expectancy:  %.2f' % ((tradeWins/numberTrades)*(tradeWinsValue/tradeWins)-(tradeLosses/numberTrades)*(tradeLossesValue/tradeLosses)))
print("Fixed trade size: ", fixedTradeDollars)

# Sharpe ratio...probably not correct math
#import math
#print(np.mean(tradeGainDollars))
#print(np.std(tradeGainDollars))
#print(math.sqrt(numberTradeDays)*(np.mean(tradeGainDollars)/np.std(tradeGainDollars)))


####  end  ####     
        
        