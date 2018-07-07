# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:40:08 2018

@author: kruegkj
"""

"""
Figure 9.11 V3
ReadSignalsCSV.py

Revision May 1, 2017
... Converted to Python 3

This program is included in the book
"Quantitative Technical Analysis"
written by Dr. Howard Bandy
and published by Blue Owl Press, Inc.
Copyright 2014 Howard Bandy

Author: Howard Bandy
Blue Owl Press, Inc.
www.BlueOwlPress.com

This program is intended to be an educational tool.
It has not been reviewed.
It is not guaranteed to be error free.
Use of any kind by any person or organization
is with the understanding that the program is as is,
including any and all of its faults.
It is provided without warranty of any kind.
No support for this program will be provided.
It is not trading advice.
The programming is intended to be clear,
but not necessarily efficient.

Permission to use, copy, and share is granted.
Please include author credits with any distribution.

"""
#%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#  Set the path for the csv file
#  You may want to specify the complete path
#  For this example, the data file is in default directory
path = 'oos_equity_eval_XLE_test1.csv'

#  Use pandas to read the csv file, 
#  creating a pandas dataFrame
sst = pd.read_csv(path)
print ("\ntype(sst): ", type(sst))

#  Print the column labels
print (sst.columns.values)
print (sst.head())
print (sst.tail())

#  Count the number of rows in the file
nrows = sst.shape[0]
print ('There are %0.f rows of data' % nrows)

#  Compute cumulative equity for all days
equityAllSignals = np.zeros(nrows)
equityAllSignals[0] = 1
for i in range(1,nrows):
    equityAllSignals[i] = (1+sst.gainAhead[i])*equityAllSignals[i-1]

print ('TWR for all signals is %0.3f' % equityAllSignals[nrows-1])
    
#  Compute cumulative equity for days with beLong signals    
equityBeLongSignals = np.zeros(nrows)
equityBeLongSignals[0] = 1
for i in range(1,nrows):
    if (sst.signal[i] > 0):
        equityBeLongSignals[i] = (1+sst.gainAhead[i])*equityBeLongSignals[i-1]
    else:
        equityBeLongSignals[i] = equityBeLongSignals[i-1]
        
print ('TWR for all days with beLong signals is %0.3f' % equityBeLongSignals[nrows-1])

#  Plot the two equity streams
plt.plot(equityBeLongSignals, '.')
plt.plot(equityAllSignals, '--')
plt.show()

####  end  ####