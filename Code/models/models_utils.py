# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:24:25 2018

@author: KRUEGKJ
"""
import matplotlib.pylab as plt
from matplotlib import cm as cm
import numpy as np
import pandas as pd

# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.
# from http://code.activestate.com/recipes/410692/
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def cumulEquity(valData, tradesData, ceType):
    nrows = tradesData.shape[0]
    #  Compute cumulative equity for all days
    eqty_signals = np.zeros(nrows)
    eqty_signals[0] = 1
    for i in range(1,nrows):
        for case in switch(ceType):
            if case('All'):
                eqty_signals[i] = (1+tradesData.gainAhead[i])*eqty_signals[i-1]
                break
            if case('BeLong'):
                if (tradesData.beLong[i] > 0):
                    eqty_signals[i] = (1+tradesData.gainAhead[i])*eqty_signals[i-1]
                else:
                    eqty_signals[i] = eqty_signals[i-1]
                break
            if case('ValBeLong'):
                if (tradesData.valBeLong[i] > 0):
                    eqty_signals[i] = (1+tradesData.gainAhead[i])*eqty_signals[i-1]
                else:
                    eqty_signals[i] = eqty_signals[i-1]
                      
    print ('TWR for ' + ceType + ' is ' + str(eqty_signals[nrows-1]))    
    var_name = 'equity' + ceType + 'Signals'        
    valData[var_name] = pd.Series(eqty_signals, index=valData.index)
    

def plotPriceAndBeLong(issue, modelStartDate, modelEndDate, valData):
    from Code.lib.plot_utils import PlotUtility
    plotIt = PlotUtility()
    plotTitle = "Close " + str(modelStartDate) + " to " + str(modelEndDate)
    plotIt.plot_v2x(valData, plotTitle)
    plt.show(block=False)
    plotIt.histogram(valData['beLong'], x_label="beLong signal", y_label="Frequency", 
      title = "beLong distribution for " + issue)        
    plt.show(block=False)

def plotPriceAndTradeSignals(valData):
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.05)
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan=3, colspan=1)
    ax2 = plt.subplot2grid((8,1), (3,0), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((8,1), (4,0), rowspan=1, colspan=1)
    
    ax2.plot(valData['valBeLong'], color='green', alpha =0.6)
    ax1.plot(valData['Close'])
    ax3.plot(valData['beLong'], color='purple', alpha =0.6)
    ax1.set_title('Comparison of beLong vs. Predicted beLong')
    ax1.label_outer()
    ax2.label_outer()
    ax2.tick_params(axis='x',which='major',bottom=True, rotation=45)
    fig.autofmt_xdate()
    axes = [ax1, ax2, ax3]
    for x in range(len(axes)):
        axes[x].grid(True, which='major', color='k', linestyle='-', alpha=0.6)
        axes[x].grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
        axes[x].minorticks_on()
        axes[x].legend(loc='upper left', frameon=True, fontsize=10)
        axes[x].label_outer()
    plt.show(block=True)
    
def plotPriceAndCumulEquity(issue, valData):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.plot(valData.equityBeLongSignals, color='green',label='BeLong')
    #ax1.plot(valData.equityAllSignals, color='blue',label='Equity')
    ax1.plot(valData.equityValBeLongSignals, color='purple',label='ValBeLong')
    
    ax1.legend(loc='upper left', frameon=True, fontsize=8)
    ax1.label_outer()
    ax1.tick_params(axis='x',which='major',bottom=True)
    ax1.minorticks_on()
    ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
    ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
    ax1.set_title('TWR for beLong, and valBeLong')
    ax2 = ax1.twinx()
    ax2.plot(valData.Close, color='black',alpha=0.6,label='CLOSE',linestyle='--')
    ax2.legend(loc='center left', frameon=True, fontsize=8)
    ax2.label_outer()
    fig.autofmt_xdate()
    plt.show(block=0)
    