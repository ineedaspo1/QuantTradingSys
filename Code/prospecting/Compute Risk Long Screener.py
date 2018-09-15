# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:09:34 2018

@author: KRUEGKJ
"""

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import seaborn as sns
import random
from scipy import stats
import time
from datetime import *
import sys
import os.path
import math
from pandas_datareader._utils import RemoteDataError


accuracy_tolerance = 0.005

# ------------------------
#   User sets parameter values here
#   Scalars, unless otherwise noted

issue_list = ['AAXJ','ACWI','ACWX','AGG','AMJ','AMLP','AMZA','ANGL','ARKK','ASHR','BIL','BIV','BKLN','BND','BNDX','BOTZ','BRZU','BSV','BWX','CIU','COMT','CSJ','CWB','DBA','DBC','DBEF','DBEU','DBJP','DBO','DDM','DGAZ','DGRO','DIA','DOG','DRIP','DUST','DVY','DXD','DXJ','EDZ','EELV','EEM','EEMO','EEMV','EFA','EFAV','EFG','EFV','EIDO','EMB','EMLC','EMLP','EPI','EPOL','EPP','ERUS','ERX','ERY','EUFN','EUM','EWA','EWC','EWD','EWG','EWH','EWI','EWJ','EWL','EWM','EWP','EWQ','EWS','EWT','EWU','EWW','EWY','EWZ','EZA','EZU','FAS','FAZ','FDN','FEZ','FLOT','FLRN','FNCL','FNDF','FNDX','FPE','FTEC','FTSM','FV','FVD','FXE','FXI','FXY','GASL','GDX','GDXJ','GLD','GOVT','GREK','GSG','GSLC','GSY','GUNR','GUSH','HDGE','HDV','HEDJ','HEFA','HEWG','HEWJ','HEZU','HYD','HYG','HYS','IAU','IBB','IDV','IEF','IEFA','IEI','IEMG','IEV','IGF','IJH','IJR','ILF','INDA','INDY','ITA','ITB','ITM','ITOT','IUSG','IUSV','IVE','IVV','IVW','IWB','IWD','IWF','IWM','IWN','IWO','IWP','IWS','IWV','IXUS','IYE','IYF','IYR','IYW','IYZ','JDST','JNK','JNUG','KBE','KBWB','KIE','KRE','KWEB','LABD','LABU','LIT','LQD','MBB','MCHI','MDY','MINT','MLPA','MLPI','MTUM','MUB','NEAR','NOBL','NUGT','OEF','OIH','OIL','PCY','PDBC','PFF','PGX','PHB','PHYS','PSLV','PSQ','QID','QLD','QQQ','QUAL','ROBO','RSP','RSX','RUSL','RWX','SCHA','SCHB','SCHD','SCHE','SCHF','SCHG','SCHH','SCHO','SCHV','SCHX','SCHZ','SCO','SCZ','SDOW','SDS','SDY','SH','SHM','SHV','SHY','SHYG','SJNK','SLV','SMH','SOXL','SOXS','SOXX','SPHD','SPLV','SPXL','SPXS','SPXU','SPY','SPYG','SPYV','SQQQ','SRLN','SRTY','SSO','SVXY','TBF','TBT','TECL','TFI','TIP','TLT','TMF','TMV','TNA','TQQQ','TVIX','TWM','TZA','UCO','UDOW','UGAZ','UGLD','UNG','UPRO','URA','USLV','USMV','USO','UUP','UVXY','VB','VBR','VCIT','VCSH','VDE','VEA','VEU','VFH','VGK','VGSH','VGT','VIG','VIXY','VLUE','VMBS','VNM','VNQ','VNQI','VO','VOO','VPL','VT','VTEB','VTI','VTIP','VTV','VUG','VV','VWO','VXUS','VXX','VYM','XBI','XES','XHB','XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XME','XOP','XRT','YINN']
"""
stocks ['AAL','AAPL','ABBV','ABEV','ABT','ABX','ACN','ADBE','ADM','ADP','AEO','AEP','AES','AFL','AGG','AGN','AIG','AKS','AMAT','AMD','AMGN','AMJ','AMLP','AMZN','ANF','APA','APC','AR','ARI','ATVI','AU','AUY','AVGO','AVP','AXP','AZN','BA','BABA','BAC','BAX','BBD','BBT','BBVA','BBY','BCS','BEN','BIDU','BK','BKD','BKLN','BMY','BP','BRK.B','BRX','BSX','BTG','BX','C','CAG','CAH','CAT','CBL','CBS','CCI','CCL','CDE','CELG','CENX','CERN','CF','CFG','CHK','CI','CIEN','CL','CLF','CLNS','CMCSA','CMS','CNP','CNX','COG','COP','COTY','CPE','CRM','CRZO','CSCO','CSX','CTL','CTRP','CTSH','CUZ','CVE','CVS','CVX','CX','CY','CZR','D','DAL','DB','DDR','DG','DHI','DIA','DIS','DISCA','DISCK','DKS','DLTR','DNR','DUK','DVN','DXD','DYN','EBAY','ECA','ECYT','EEM','EEP','EFA','EGO','EIX','EMR','ENB','ENDP','EOG','EPD','EQR','ERIC','ESRX','ESV','ETE','ETN','ETP','EWC','EWG','EWH','EWJ','EWT','EWW','EWY','EWZ','EXC','F','FB','FCAU','FCX','FDC','FE','FEYE','FEZ','FHN','FIT','FITB','FL','FLEX','FNB','FNSR','FOX','FOXA','FTI','FXI','GDX','GDXJ','GE','GERN','GFI','GG','GGB','GGP','GILD','GIS','GLD','GLW','GM','GME','GNW','GPOR','GPS','GRPN','GSK','HAL','HBAN','HBI','HCP','HD','HES','HIMX','HK','HL','HMY','HPE','HPQ','HRB','HRL','HST','HUN','HYG','IAG','IAU','IBM','IBN','IJR','IMGN','INFY','ING','INTC','IP','IPG','ITUB','IVV','IVZ','IWM','IYR','JBLU','JCI','JCP','JD','JNJ','JNK','JNPR','JPM','KEY','KGC','KHC','KIM','KKR','KMI','KO','KR','KRE','KSS','LB','LC','LEN','LLY','LOW','LQD','LRCX','LUV','LVS','LYG','M','MA','MAS','MAT','MCD','MCHI','MDLZ','MDR','MDT','MET','MFA','MFGP','MGM','MNK','MNST','MO','MOMO','MON','MOS','MPC','MRK','MRO','MRVL','MS','MSCC','MSFT','MT','MTG','MU','MULE','MYL','NBL','NBR','NE','NEM','NFLX','NFX','NI','NKE','NKTR','NLY','NOK','NRG','NTNX','NUGT','NVAX','NVDA','NWL','NXPI','NYCB','OAS','OCLR','ODP','OIH','OKE','ON','OPK','ORCL','OSTK','OXY','P','PAA','PAH','PBCT','PBR','PBR.A','PCAR','PCG','PDCO','PE','PEG','PEP','PFE','PG','PHM','PK','PLUG','PM','PPL','PSTG','PTEN','PTI','PYPL','QCOM','QD','QEP','QID','QQQ','RAD','RDS.A','RF','RIG','RIO','ROKU','RRC','RSPP','RSX','S','SAN','SBGL','SBUX','SCHW','SDOW','SDS','SGYP','SH','SIG','SIRI','SLB','SLM','SLV','SM','SMH','SNAP','SO','SPXL','SPXS','SPXU','SPY','SQ','SQQQ','SRC','SRNE','SSO','STNG','STX','SVXY','SWN','SYF','SYMC','SYY','T','TAL','TBT','TEVA','TGT','TJX','TLT','TMUS','TNA','TQQQ','TRQ','TSLA','TSM','TTWO','TV','TVIX','TWTR','TWX','TXN','TZA','UA','UAL','UNH','UNP','UPS','USB','USO','UTX','UVXY','V','VALE','VEA','VEON','VER','VGK','VIAB','VICI','VIPS','VLO','VLY','VNQ','VOD','VOO','VRX','VTI','VTR','VWO','VXX','VZ','WBA','WDC','WEN','WFC','WFT','WIN','WLL','WMB','WMT','WPX','WU','WY','WYNN','X','XBI','XEL','XL','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XOM','XOP','XRT','ZNGA','ZTO']
"""

"""
ETFs ['AAXJ','ACWI','ACWX','AGG','AMJ','AMLP','AMZA','ANGL','ARKK','ASHR','BIL','BIV','BKLN','BND','BNDX','BOTZ','BRZU','BSV','BWX','CIU','COMT','CSJ','CWB','DBA','DBC','DBEF','DBEU','DBJP','DBO','DDM','DGAZ','DGRO','DIA','DOG','DRIP','DUST','DVY','DWT','DXD','DXJ','EDZ','EELV','EEM','EEMO','EEMV','EFA','EFAV','EFG','EFV','EIDO','EMB','EMLC','EMLP','EPI','EPOL','EPP','ERUS','ERX','ERY','EUFN','EUM','EWA','EWC','EWD','EWG','EWH','EWI','EWJ','EWL','EWM','EWP','EWQ','EWS','EWT','EWU','EWW','EWY','EWZ','EZA','EZU','FAS','FAZ','FDN','FEZ','FLOT','FLRN','FNCL','FNDF','FNDX','FPE','FTEC','FTSM','FV','FVD','FXE','FXI','FXY','GASL','GDX','GDXJ','GLD','GOVT','GREK','GSG','GSLC','GSY','GUNR','GUSH','HDGE','HDV','HEDJ','HEFA','HEWG','HEWJ','HEZU','HYD','HYG','HYLB','HYS','IAU','IBB','IDV','IEF','IEFA','IEI','IEMG','IEV','IGF','IJH','IJR','ILF','INDA','INDY','ITA','ITB','ITM','ITOT','IUSG','IUSV','IVE','IVV','IVW','IWB','IWD','IWF','IWM','IWN','IWO','IWP','IWS','IWV','IXUS','IYE','IYF','IYR','IYW','IYZ','JDST','JNK','JNUG','KBE','KBWB','KIE','KRE','KWEB','LABD','LABU','LIT','LQD','MBB','MCHI','MDY','MINT','MLPA','MLPI','MTUM','MUB','NEAR','NOBL','NUGT','OEF','OIH','OIL','PCY','PDBC','PFF','PGX','PHB','PHYS','PSLV','PSQ','QID','QLD','QQQ','QUAL','ROBO','RSP','RSX','RUSL','RWX','SCHA','SCHB','SCHD','SCHE','SCHF','SCHG','SCHH','SCHO','SCHV','SCHX','SCHZ','SCO','SCZ','SDOW','SDS','SDY','SH','SHM','SHV','SHY','SHYG','SJNK','SLV','SMH','SOXL','SOXS','SOXX','SPAB','SPDW','SPEM','SPHD','SPIB','SPLG','SPLV','SPSB','SPSM','SPTM','SPXL','SPXS','SPXU','SPY','SPYG','SPYV','SQQQ','SRLN','SRTY','SSO','SVXY','TBF','TBT','TECL','TFI','TIP','TLT','TMF','TMV','TNA','TQQQ','TVIX','TWM','TZA','UCO','UDOW','UGAZ','UGLD','UNG','UPRO','URA','USLV','USMV','USO','UUP','UVXY','UWT','VB','VBR','VCIT','VCSH','VDE','VEA','VEU','VFH','VGK','VGSH','VGT','VIG','VIXY','VLUE','VMBS','VNM','VNQ','VNQI','VO','VOO','VPL','VT','VTEB','VTI','VTIP','VTV','VUG','VV','VWO','VXUS','VXX','VYM','XBI','XES','XHB','XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XME','XOP','XRT','YINN']
"""

issues = []
issue_output_list = []

data_source = 'morningstar'
start_date = datetime(2000,1,1)
end_date = datetime(2014,1,1)

hold_days = 3
system_accuracy = 0.6
DD95_limit = 0.15
initial_equity = 100000.0
fraction = 1.00
forecast_horizon = 2*252 #  trading days
number_forecasts = 20  # Number of simulated forecasts

print ('\n\nNew simulation run ')
print ('  Testing profit potential for Long positions\n ')
print ('Dates:       ' + start_date.strftime('%d %b %Y')) 
print (' to:         ' + end_date.strftime('%d %b %Y'))
print ('Hold Days:               %i ' % hold_days)
print ('System Accuracy:         %.2f ' % system_accuracy)
print ('DD 95 limit:        %.2f ' % DD95_limit)
print ('Forecast Horizon:   %i ' % forecast_horizon)
print ('Number Forecasts:   %i ' % number_forecasts)
print ('Initial Equity:     %i ' % initial_equity)

# ------------------------
#  Variables used for simulation

# Loop through tickers
#tickers = ['IBM','AAPL']
#df = pd.concat([web.DataReader(ticker,'morningstar', start, end) for ticker in tickers]).reset_index()

for issue in issue_list:
    print ('===================================')
    print ('Issue: ' + issue)

    # Download data
    try:
        qt = pdr.DataReader(issue, data_source, start_date, end_date).reset_index()
    except RemoteDataError:
        print("No information for ticker '%s'" % issue)
        continue
    except TypeError:
        continue
    except:
        continue
    print (qt.shape)
    print (qt.head())
    
    latest_date = qt["Date"].iloc[0]
    earliest_date = qt["Date"].iloc[-1]
    difference_in_years = relativedelta(earliest_date, latest_date).years
    
    qt = qt.drop("Symbol", axis =1)
    qt.set_index('Date', inplace=True)
    
    #qt['Close'].plot(figsize=(13,7))
    
     
    nrows = qt.shape[0]
    print ('Number Rows:        %d ' % nrows)
    
    qtC = qt.Close
    
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
        # print ('Using fraction: %.3f ' % fraction,)
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
        #print ('  DD95: %.3f ' % DD_95)
        #print ('  Equity: %.2f ' % MTM_equity)
        if (abs(DD95_limit - DD_95) < accuracy_tolerance):
            #  Close enough 
            done = True
        else:
            #  Adjust fraction and make a new set of forecasts
            fraction = fraction * DD95_limit / DD_95
            done = False
            
    #  Report 
    #IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
    #IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)
    print ('Using fraction: %.3f ' % fraction,)        
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
    temp_output = [issue,fraction,IT_DD_95,CAR_25,CAR_50,CAR_75,difference_in_years]
    #print(temp_output)
    issue_output_list.append(temp_output)

# Save droawdown output 
#print(issue_output_list)
filenameext = '_holddays' + str(hold_days) + '_accuracy' + str(system_accuracy) + '_DD95limit' + str(DD95_limit) + '_' +\
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".csv"    
filename = 'etf screener_fraction' + filenameext
#np.savetxt(filename, issue_output_list, delimiter=',',header="Issue,fraction,DD95,CAR25,CAR50,CAR75", fmt='%20s %16.2f %16.2f %16.2f %16.2f %16.2f')
np.savetxt(filename, issue_output_list, delimiter=',',header="Issue,fraction,DD95,CAR25,CAR50,CAR75,diff yrs", fmt='%20s')

####  end  ####