# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:11:33 2019

@author: kruegkj
"""
import numpy as np
from scipy import stats
import datetime as dt

class TradeRisk:
    
    def get_safef_car25(self, sst, iStart, iEnd, tms_dict):
        
        forecastHorizon = tms_dict["forecastHorizon"]
        initialEquity = tms_dict["initialEquity"]
        ddTolerance = tms_dict["ddTolerance"]
        tailRiskPct = tms_dict["tailRiskPct"]
        windowLength = tms_dict["windowLength"]
        nCurves = tms_dict["nCurves"]
        updateInterval = tms_dict["updateInterval"]
        print(tms_dict)
        
        years_in_forecast = forecastHorizon / 252.0
        
        printDetails = False
        
        for i in range(iStart, iEnd+1, updateInterval):
            if printDetails: 
                print ("\nDate: ", dt.datetime.strftime(sst.index[i], '%Y-%m-%d'))
                print ("beLong: ", sst.signal[i])
                print ("gain Ahead: {0:.4f}".format(sst.gainAhead[i]))
        
            #  Initialize variables
            curves = np.zeros(nCurves)
            numberDraws = np.zeros(nCurves)
            TWR = np.zeros(nCurves)
            maxDD = np.zeros(nCurves)
            
            fraction = 1.00
            dd95 = 2 * ddTolerance
            
            while (abs(dd95-ddTolerance)>0.03):
                #  Generate nCurve equity curves
                if printDetails: 
                    print  ("    Fraction {0:.2f}".format(fraction))
        #    
                for nc in range(nCurves):
                    #print ("working on curve ", nc)
                    equity = initialEquity
                    maxEquity = equity
                    drawdown = 0
                    maxDrawdown = 0
                    horizonSoFar = 0
                    nd = 0
                    while (horizonSoFar < forecastHorizon):
                        j = np.random.randint(0,windowLength)
                #        print j
                        nd = nd + 1
                        weightJ = 1.00 - j/windowLength
                #        print weightJ
                        horizonSoFar = horizonSoFar + weightJ
                        signalJ = sst.signal[i-j]
                        if signalJ > 0:
                            tradeJ = sst.gainAhead[i-j] * weightJ
                        else:
                            tradeJ = 0.0
                        thisTrade = fraction * tradeJ * equity    
                        equity = equity + thisTrade
                        maxEquity = max(equity,maxEquity)
                        drawdown = (maxEquity-equity)/maxEquity
                        maxDrawdown = max(drawdown,maxDrawdown)
            #        print "equity, maxDD, ndraws:", equity, maxDrawdown, nd        
                    TWR[nc] = equity
                    maxDD[nc] = maxDrawdown
                    numberDraws[nc] = nd
            
                #  Find the drawdown at the tailLimit-th percentile        
                dd95 = stats.scoreatpercentile(maxDD,tailRiskPct)
                if printDetails: 
                    print ('  DD {0}: {1:.3f} '.format(tailRiskPct, dd95))
                fraction = fraction * ddTolerance / dd95
                TWR25 = stats.scoreatpercentile(TWR,25)        
                CAR25 = 100*(((TWR25/initialEquity) ** (1.0/years_in_forecast))-1.0)
            if printDetails: 
                print ('Fraction: {0:.2f}'.format(fraction))
                print ('CAR25: {0:.2f}'.format(CAR25))
            sst.iloc[i,sst.columns.get_loc('safef')] = fraction
            sst.iloc[i,sst.columns.get_loc('CAR25')] = CAR25
            #sst.loc[i,'CAR25'] = CAR25
        return sst
        
    def update_tms_trade_dec(self, tms, iStart, iEnd, y_validate):
        # Update trade_decision with current date decision
        if y_validate[0] == 1: 
            if tms.iloc[iEnd,tms.columns.get_loc('CAR25')] > 10:
                tms.iloc[iEnd,tms.columns.get_loc('trade_decision')] = 'Long'
            else:
                tms.iloc[iEnd,tms.columns.get_loc('trade_decision')] = 'Flat'
        else:
            tms.iloc[iEnd,tms.columns.get_loc('trade_decision')] = 'Flat'
        temp_tms = self.update_tms(tms, iStart, iEnd)
        return temp_tms
     
    def update_tms(self, tms, iStart, iEnd):
        for i in range(iStart, iEnd):
            if tms.trade_decision[i] == 'Long':
                tms.iloc[i,tms.columns.get_loc('trade')] = tms.iloc[i-1,tms.columns.get_loc('fract')] * tms.iloc[i-1,tms.columns.get_loc('equity')] * tms.iloc[i,tms.columns.get_loc('gainAhead')]
            elif tms.signal[i] > 0:
                print('signal > 0')
                if tms.CAR25[i] > 10:
                    print('CAR25 > 10')
                    temp = tms.iloc[i-1,tms.columns.get_loc('fract')] * tms.iloc[i-1,tms.columns.get_loc('equity')] * tms.iloc[i,tms.columns.get_loc('gainAhead')]
                    print(temp)
                    tms.iloc[i,tms.columns.get_loc('trade')] = tms.iloc[i-1,tms.columns.get_loc('fract')] * tms.iloc[i-1,tms.columns.get_loc('equity')] * tms.iloc[i,tms.columns.get_loc('gainAhead')]
            else:
                print('trade = 0')
                tms.iloc[i,tms.columns.get_loc('trade')] = 0.0
                
            tms.iloc[i,tms.columns.get_loc('fract')] = tms.iloc[i,tms.columns.get_loc('safef')]
            
            tms.iloc[i,tms.columns.get_loc('equity')] = tms.iloc[i-1,tms.columns.get_loc('equity')] + tms.iloc[i,tms.columns.get_loc('trade')]
            tms.iloc[i,tms.columns.get_loc('maxEquity')] = max(tms.iloc[i,tms.columns.get_loc('equity')],tms.iloc[i-1,tms.columns.get_loc('maxEquity')])
            tms.iloc[i,tms.columns.get_loc('drawdown')] = (tms.iloc[i,tms.columns.get_loc('maxEquity')] - tms.iloc[i,tms.columns.get_loc('equity')]) / tms.iloc[i,tms.columns.get_loc('maxEquity')]
            tms.iloc[i,tms.columns.get_loc('maxDD')] =  max(tms.iloc[i,tms.columns.get_loc('drawdown')],tms.iloc[i-1,tms.columns.get_loc('maxDD')])
            tms.iloc[i,tms.columns.get_loc('fract')] = tms.iloc[i,tms.columns.get_loc('safef')]
            
        return tms
    
class PriceUpdate:
    ''' For new price entry, provide menu to:
        1. Skip entry (holiday,, no trading day)
        2. Manual entry
        3. Automatic EOD
        4. Automatic Intrday
        '''
    def nothing_yet():
        hi = ""
        
