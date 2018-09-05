# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
oscillator_studies.py
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class OscialltorStudies:
    """Group of oscillator-based indicator features"""
    def detrend_PO(self, p, lb, feature_dict):
        """<add content here>
           Args:
                p: data series
                lb: number of days to look back
                feature_dict: Dictionary of added features
           Return:
                d: numpy array with values centered on 0.0.
                feature_dict
        """
        feature_dict['DPO_' + str(lb)] = 'Keep'
        nrows = p.shape[0]
        ma = p.ewm(span=lb,
                   min_periods=0,
                   adjust=True,
                   ignore_na=False).mean()
        d = np.zeros(nrows)
        for i in range(1, nrows):
            d[i] = (p[i] - ma[i]) / ma[i]

        return d, feature_dict
"""
#    def add_indicators(self,df, ind_list, feature_dict):
#        # loop through ind_list
#        i = 0
#        for i in ind_list:
#            print(i)
#            sel_ind = i[0]
#            if sel_ind == 'RSI':
#                df['RSI'],feature_dict = self.RSI(df.Pri,
#                                                  i[1],
#                                                  feature_dict
#                                                  )
#            elif sel_ind == 'DPO':
#                df['DPO'],feature_dict = self.detrend_PO(df.Pri,
#                                                         i[1],
#                                                         feature_dict
#                                                         )
#            elif sel_ind == 'ROC':
#                df['ROC'],feature_dict = self.ROC(df.Pri,
#                                                  i[1],
#                                                  feature_dict
#                                                  )
#            else:
#                continue
#        return df, feature_dict
"""

if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}

    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)

    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )

    oscSt = OscialltorStudies()

    dataSet['DPO_10'], feature_dict = oscSt.detrend_PO(dataSet.Pri,
                                                       10,
                                                       feature_dict
                                                       )
    dataSet['DPO_20'], feature_dict = oscSt.detrend_PO(dataSet.Pri,
                                                       20,
                                                       feature_dict
                                                       )

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plot_df = dataSet.ix[startDate:endDate]

    fig, axes = plt.subplots(3, 1,
                             figsize=(15, 8),
                             sharex=True
                             )

    N = len(plot_df)
    ind = np.arange(N)  # the evenly spaced plot indices

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return plot_df.Date[thisind].strftime('%Y-%m-%d')

    axes[0].plot(ind, plot_df['Pri'], label=issue)
    axes[1].plot(ind, plot_df['DPO_10'], label='DPO_10')
    axes[2].plot(ind, plot_df['DPO_20'], label='DPO_20')

    plt.subplots_adjust(hspace=0)
    for ax in axes:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.grid(True, which='both')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', bottom='off')
        fig.autofmt_xdate()
