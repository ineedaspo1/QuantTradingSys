# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:34:16 2018
@author: kruegkj
plot_utils.py
"""

import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np

class PlotUtility:

    def plot_v1(self, data, title):
        fig, ax = plt.subplots(figsize=(10,4))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
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
        
    def plot_v2x(self, plotDataSet, title):
        numSubPlots = 2
        fig, axes = plt.subplots(numSubPlots, ncols=1, figsize=(numSubPlots*6,8), sharex=True)
        buys = plotDataSet.ix[(plotDataSet['beLong'] > 0)]
        sells = plotDataSet.ix[(plotDataSet['beLong'] < 0)]
        
        axes[0].plot(plotDataSet.index, plotDataSet['Close'])
        axes[0].plot(buys.index, plotDataSet.ix[buys.index]['Close'], '^', markersize=10, color='g', label='Buy')
        axes[0].plot(sells.index, plotDataSet.ix[sells.index]['Close'], 'v', markersize=10, color='r', label='Sell')
        axes[1].plot(plotDataSet['beLong'], color='red', alpha =0.8)
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
    
    def plot_beLongs(self, title, issue, df, start_date, end_date):
        plotTitle = title + ": " + issue + ", " + str(start_date) + " to " + str(end_date)
        self.plot_v2x(df, plotTitle)
        self.histogram(df['beLong'], x_label="beLong signal", y_label="Frequency", title = "beLong distribution for " + issue)
        
    def price_Ind_Vol_Plot(self, plot_dict, df):
        # Subplots are organized in a Rows x Cols Grid
        issue = plot_dict['Issue']
        key_to_value_lengths = {k:len(v) for k, v in plot_dict.items()}
        #print(key_to_value_lengths)
        #key_to_value_lengths['Plot_Vars']
        subplot_len = key_to_value_lengths['Plot_Vars']
        #print(subplot_len)
        if plot_dict['Volume']=='Yes':
            total_rows = 2 + subplot_len
        else:
            total_rows = 1 + subplot_len
    
        Cols = 1
        
        N = len(df)
        ind = np.arange(N)  # the evenly spaced plot indices
    
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, N - 1)
            return df.Date[thisind].strftime('%Y-%m-%d')
        
        fig = plt.figure(1,figsize=(14,total_rows*2))
        plt.subplots_adjust(hspace=0.05)
        
        cnt = 0
        for n in range(1,total_rows+1):
            if n==1:
                ax = fig.add_subplot(total_rows,Cols,1)
                ax.plot(ind, df['Close'], label=issue)
            elif n < subplot_len+2:
                ax = fig.add_subplot(total_rows,Cols,n,sharex=ax)
                ax.plot(ind, df[plot_dict['Plot_Vars'][cnt]], label=plot_dict['Plot_Vars'][cnt])
                cnt += 1
            else: # add Volume plot if requested
                ax = fig.add_subplot(total_rows,Cols,n)
                ax.bar(ind, df['Volume'], label='Volume')
            
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            ax.label_outer()
            ax.legend(loc='upper left', frameon=True, fontsize=10)
            ax.minorticks_on()
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        plt.show()
        

if __name__ == "__main__":
    plotIt = PlotUtility()
    dataLoadStartDate = "2008-02-01"
    dataLoadEndDate = "2010-04-01"
    issue = "XLV"
    
    from retrieve_data import *
    dSet = DataRetrieve()
    
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    beLongThreshold = 0
    cT = ComputeTarget()
    dataSet = cT.setTarget(dataSet, 
                           "Long", 
                           beLongThreshold
                           )
    
    # Plot price and indicators
    startDate = "2008-02-01"
    endDate = "2010-04-01"

    plotDataSet = dataSet[startDate:endDate].copy()

    # Set up plot dictionary
    plot_dict = {}
    plot_dict['Issue'] = dataSet.Symbol[0]
    plot_dict['Plot_Vars'] = ['beLong']
    plot_dict['Volume'] = 'Yes'
    
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDataSet)
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(plotDataSet['Close'], plotTitle)

        
    plotTitle = "beLong signal for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(plotDataSet['beLong'], plotTitle)
    
    plotIt.histogram(
            plotDataSet['beLong'], 
            x_label="beLong signal", 
            y_label="Frequency", 
            title = "beLong distribution for " + issue)
    
    plotTitle = issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v2x(plotDataSet, plotTitle)
    
    plotIt.plot_beLongs("Plot of beLongs", issue, plotDataSet, dataLoadStartDate, dataLoadEndDate)
    
