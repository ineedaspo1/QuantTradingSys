# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:09:34 2018

@author: KRUEGKJ
"""

from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import random
from scipy import stats
import time
from datetime import *
import sys

accuracy_tolerance = 0.005

# ------------------------
#   User sets parameter values here
#   Scalars, unless otherwise noted

issue = 'spy'
data_source = 'morningstar'
start_date = datetime(1999,1,1)
end_date = datetime(2012,1,1)

#hold_days = 5
hold_days_list = [1, 2, 3, 5, 10, 21, 63, 126, 252]
system_accuracy = 0.5
DD95_limit = 0.15
initial_equity = 100000.0
fraction = 1.00
forecast_horizon = 504 #  trading days
number_forecasts = 20  # Number of simulated forecasts

print ('\n\nNew simulation run ')
print ('  Testing profit potential for Long positions\n ')
print ('Issue:              ' + issue)
print ('Dates:       ' + start_date.strftime('%d %b %Y')) 
print (' to:         ' + end_date.strftime('%d %b %Y'))
#print ('Hold Days:               %i ' % hold_days)
print ('System Accuracy:         %.2f ' % system_accuracy)
print ('DD 95 limit:        %.2f ' % DD95_limit)
print ('Forecast Horizon:   %i ' % forecast_horizon)
print ('Number Forecasts:   %i ' % number_forecasts)
print ('Initial Equity:     %i ' % initial_equity)

# ------------------------
#  Variables used for simulation

qt = pdr.DataReader(issue, data_source, start_date, end_date)
print (qt.shape)
print (qt.head())

nrows = qt.shape[0]
print ('Number Rows:        %d ' % nrows)

qtC = qt.Close

# loop through hold days
for hold_days in hold_days_list:
    print ('##########################')
    print ('Number of Hold Days:        %i ' % hold_days)
    system_accuracy = 0.5
    number_trades = int(forecast_horizon / hold_days)
    number_days = number_trades*hold_days
    print ('Number Days:        %i ' % number_days)
    print ('Number Trades:      %d ' % number_trades)
    
    al = int(number_days+1)
    #   These arrays are the number of days in the forecast
    account_balance = np.zeros(al)     # account balance
    
    pltx = np.zeros(al)
    plty = np.zeros(al)
    
    max_IT_DD = np.zeros(al)     # Maximum Intra-Trade drawdown
    max_IT_Eq = np.zeros(al)     # Maximum Intra-Trade equity
    
    #   These arrays are the number of simulation runs
    # Max intra-trade drawdown
    FC_max_IT_DD = np.zeros(number_forecasts)  
    # Trade equity (TWR)
    FC_tr_eq = np.zeros(number_forecasts)  
    
    # ------------------------
    #   Set up gainer and loser lists
    gainer = np.zeros(nrows)
    loser = np.zeros(nrows)
    i_gainer = 0
    i_loser = 0
    
    for i in range(0,nrows-hold_days):
        if (qtC[i+hold_days]>qtC[i]):
            gainer[i_gainer] = i
            i_gainer = i_gainer + 1
        else:
            loser[i_loser] = i
            i_loser = i_loser + 1
    number_gainers = i_gainer
    number_losers = i_loser
    
    print ('Number Gainers:     %d ' % number_gainers)
    print ('Number Losers:      %d ' % number_losers)
    
    #################################################
    #  Solve for fraction
    fraction = 1.00
    done = False
    
    while not done:
        done = True
        print ('Using fraction: %.3f ' % fraction,)
        # -----------------------------
        #   Beginning a new forecast run
        for i_forecast in range(number_forecasts):
        #   Initialize for trade sequence
            i_day = 0    # i_day counts to end of forecast
            #  Daily arrays, so running history can be plotted
            # Starting account balance
            account_balance[0] = initial_equity
            # Maximum intra-trade equity
            max_IT_Eq[0] = account_balance[0]    
            max_IT_DD[0] = 0
        
            #  for each trade
            for i_trade in range(0,number_trades):
                #  Select the trade and retrieve its index 
                #  into the price array
                #  gainer or loser?
                #  Uniform for win/loss
                gainer_loser_random = np.random.random()  
                #  pick a trade accordingly
                #  for long positions, test is â€œ<â€
                #  for short positions, test is â€œ>â€
                if gainer_loser_random < system_accuracy:
                    #  choose a gaining trade
                    gainer_index = random.randint(0,number_gainers)
                    entry_index = int(gainer[gainer_index])
                else:
                    #  choose a losing trade
                    loser_index = random.randint(0,number_losers)
                    entry_index = int(loser[loser_index])
                
                #  Process the trade, day by day
                for i_day_in_trade in range(0,hold_days+1):
                    if i_day_in_trade==0:
                        #  Things that happen immediately 
                        #  after the close of the signal day
                        #  Initialize for the trade
                        buy_price = qtC[entry_index]
                        number_shares = account_balance[i_day] * \
                                        fraction / buy_price
                        share_dollars = number_shares * buy_price
                        cash = account_balance[i_day] - share_dollars
                    else:
                        #  Things that change during a
                        #  day the trade is held
                        i_day = i_day + 1
                        j = entry_index + i_day_in_trade
                        #  Drawdown for the trade
                        profit = number_shares * (qtC[j] - buy_price)
                        MTM_equity = cash + share_dollars + profit
                        IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                                / max_IT_Eq[i_day-1]
                        max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                                IT_DD)
                        max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                                MTM_equity)
                        account_balance[i_day] = MTM_equity
                    if i_day_in_trade==hold_days:
                        #  Exit at the close
                        sell_price = qtC[j]
                        #  Check for end of forecast
                        if i_day >= number_days:
                            FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                            FC_tr_eq[i_forecast] = MTM_equity
                
        #  All the forecasts have been run
        #  Find the drawdown at the 95th percentile        
        DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
        print ('  DD95: %.3f ' % DD_95)
        print ('  Equity: %.2f ' % MTM_equity)
        print ('  System accuracy: %.2f ' % system_accuracy)
        """   Take fraction loop out
        if (abs(DD95_limit - DD_95) < accuracy_tolerance):
            #  Close enough 
            done = True
        else:
            #  Adjust fraction and make a new set of forecasts
            fraction = fraction * DD95_limit / DD_95
            done = False
        """
        if system_accuracy > 0.9:
            done = True
        else:
            system_accuracy += 0.05
            done = False
           
#  Report 
#IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
#IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)        
IT_DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
print ('DD95: %.3f ' % IT_DD_95,)

years_in_forecast = forecast_horizon / 252.0

TWR_25 = stats.scoreatpercentile(FC_tr_eq,25)        
CAR_25 = 100*(((TWR_25/initial_equity) ** (1.0/years_in_forecast))-1.0)
TWR_50 = stats.scoreatpercentile(FC_tr_eq,50)
CAR_50 = 100*(((TWR_50/initial_equity) ** (1.0/years_in_forecast))-1.0)
TWR_75 = stats.scoreatpercentile(FC_tr_eq,75)        
CAR_75 = 100*(((TWR_75/initial_equity) ** (1.0/years_in_forecast))-1.0)

print ('CAR25: %.2f ' % CAR_25,)
print ('CAR50: %.2f ' % CAR_50,)
print ('CAR75: %.2f ' % CAR_75)

#  Save equity curve to disc

filenameext = '_' + issue + '_holddays' + str(hold_days) + '_accuracy' + str(system_accuracy)\
            + '_DD95limit' + str(DD95_limit) + '_' +\
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".csv"
filename = 'account_balance' + filenameext        
print ("File name is "+filename)
np.savetxt(filename, account_balance, delimiter=',')

#  Save CDF data to disc
filename = 'FC_maxIT_DD' + filenameext 
np.savetxt(filename, FC_max_IT_DD, delimiter=',')
filename = 'FCTr' + filenameext 
np.savetxt(filename, FC_tr_eq, delimiter=',')

#  Plot maximum drawdown
for i in range(al):
    pltx[i] = i
    plty[i] = max_IT_DD[i]

plt.plot(pltx,plty)

####  end  ####