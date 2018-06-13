# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:34:16 2018

@author: kruegkj

plot_utils.py
"""
from  retrieve_data import *
##from retrieve_data import DataRetrieve
#from retrieve_data import ComputeTarget
##import retrieve_data

import matplotlib.pylab as plt
import matplotlib as mpl
plt.style.use('seaborn-ticks')
import matplotlib.ticker as ticker

class PlotUtility:

    def plot_v1(self, data, title):
        fig, ax = plt.subplots(figsize=(10,4))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.plot(data)
        
        # Label the axes and provide a title
        ax.set_title(title)
        ax.grid(True, which='both')
        fig.autofmt_xdate()
        ax.xaxis_date()
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()
        ax.tick_params(axis='y',which='major',bottom='off')
        return fig, ax
    
    def histogram(self, data, x_label, y_label, title):
        fig, ax = plt.subplots()
        ax.hist(data, color = '#539caf', bins = 3)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
    
        tick_spacing = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()
        ax.tick_params(axis='y',which='minor',bottom='off')
        
    def plot_v2x(self, data1, data2, title):
        numSubPlots = 2
        fig, axes = plt.subplots(numSubPlots, ncols=1, figsize=(numSubPlots*6,8), sharex=True)
        
        axes[0].plot(data1)
        axes[1].plot(data2, color='red', alpha =0.8)
        plt.subplots_adjust(hspace=0.05)
        fig.suptitle(title)
        fig.autofmt_xdate()
        for ax in axes:
            ax.label_outer()
            ax.legend(loc='upper left', frameon=True, fontsize=8)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax.grid(True, which='both')
            ax.xaxis_date()
            ax.autoscale_view()
            ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
            ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            ax.minorticks_on()
            #ax.tick_params(axis='y',which='minor',bottom='off')
        #axes[1].set_yticks((-1,0,1), minor=False)
        axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f')) 
        return fig, (axes[0], axes[1])
        
    

if __name__ == "__main__":
    dataLoadStartDate = "2017-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    plotIt = PlotUtility()
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(dataSet['Pri'], plotTitle)
    
    beLongThreshold = 0
    cT = ComputeTarget()
    mmData = cT.setTarget(dataSet, "Long", beLongThreshold)
    
    plotIt = PlotUtility()
    
    plotTitle = "beLong signal for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(mmData['beLong'], plotTitle)
    
    plotIt.histogram(mmData['beLong'], x_label="beLong signal", y_label="Frequency", 
          title = "beLong distribution for " + issue)
    
    plotTitle = issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v2x(mmData['Pri'], mmData['beLong'], plotTitle)