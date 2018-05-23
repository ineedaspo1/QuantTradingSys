# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:22:12 2018

@author: kruegkj

Read and store equity data to disk
"""
from datetime import datetime
import pandas_datareader.data as pdr
import os

from pandas_datareader._utils import RemoteDataError

# 4-14-18 Future update: move issue list to file to be read in
issue_list = ['ACWI','AGG','AMJ','AMLP','BND','BWX','DBC','DBO','DIA','DOG','DUST','DXD','EEM','EFA','EMB','EMLC','EPI','ERX','EWA','EWC','EWG','EWH','EWJ','EWL','EWQ','EWS','EWT','EWU','EWW','EWY','EWZ','EZU','FAS','FAZ','FEZ','FXI','GDX','GDXJ','GLD','HYG','IAU','IBB','IEF','IEFA','IJH','IJR','ILF','ITB','IVE','IVV','IVW','IWB','IWD','IWF','IWM','IYR','JNK','KBE','KRE','LQD','MCHI','MDY','NUGT','OIH','PCY','PFF','PGX','PSQ','QID','QLD','QQQ','RSX','SCHD','SCHF','SCHX','SCO','SCZ','SDOW','SDS','SH','SHY','SLV','SMH','SPLV','SPXL','SPXS','SPXU','SPY','SQQQ','SSO','SVXY','TBT','TIP','TLT','TMV','TNA','TQQQ','TWM','TZA','UCO','UDOW','UNG','UPRO','USO','UVXY','VCIT','VEA','VEU','VGK','VIXY','VNQ','VOO','VT','VTI','VTV','VWO','VXX','XBI','XHB','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XME','XOP','XRT','YINN']

"""
stocks ['AAL','AAPL','ABBV','ABEV','ABT','ABX','ACN','ADBE','ADM','ADP','AEO','AEP','AES','AFL','AGG','AGN','AIG','AKS','AMAT','AMD','AMGN','AMJ','AMLP','AMZN','ANF','APA','APC','AR','ARI','ATVI','AU','AUY','AVGO','AVP','AXP','AZN','BA','BABA','BAC','BAX','BBD','BBT','BBVA','BBY','BCS','BEN','BIDU','BK','BKD','BKLN','BMY','BP','BRK.B','BRX','BSX','BTG','BX','C','CAG','CAH','CAT','CBL','CBS','CCI','CCL','CDE','CELG','CENX','CERN','CF','CFG','CHK','CI','CIEN','CL','CLF','CLNS','CMCSA','CMS','CNP','CNX','COG','COP','COTY','CPE','CRM','CRZO','CSCO','CSX','CTL','CTRP','CTSH','CUZ','CVE','CVS','CVX','CX','CY','CZR','D','DAL','DB','DDR','DG','DHI','DIA','DIS','DISCA','DISCK','DKS','DLTR','DNR','DUK','DVN','DXD','DYN','EBAY','ECA','ECYT','EEM','EEP','EFA','EGO','EIX','EMR','ENB','ENDP','EOG','EPD','EQR','ERIC','ESRX','ESV','ETE','ETN','ETP','EWC','EWG','EWH','EWJ','EWT','EWW','EWY','EWZ','EXC','F','FB','FCAU','FCX','FDC','FE','FEYE','FEZ','FHN','FIT','FITB','FL','FLEX','FNB','FNSR','FOX','FOXA','FTI','FXI','GDX','GDXJ','GE','GERN','GFI','GG','GGB','GGP','GILD','GIS','GLD','GLW','GM','GME','GNW','GPOR','GPS','GRPN','GSK','HAL','HBAN','HBI','HCP','HD','HES','HIMX','HK','HL','HMY','HPE','HPQ','HRB','HRL','HST','HUN','HYG','IAG','IAU','IBM','IBN','IJR','IMGN','INFY','ING','INTC','IP','IPG','ITUB','IVV','IVZ','IWM','IYR','JBLU','JCI','JCP','JD','JNJ','JNK','JNPR','JPM','KEY','KGC','KHC','KIM','KKR','KMI','KO','KR','KRE','KSS','LB','LC','LEN','LLY','LOW','LQD','LRCX','LUV','LVS','LYG','M','MA','MAS','MAT','MCD','MCHI','MDLZ','MDR','MDT','MET','MFA','MFGP','MGM','MNK','MNST','MO','MOMO','MON','MOS','MPC','MRK','MRO','MRVL','MS','MSCC','MSFT','MT','MTG','MU','MULE','MYL','NBL','NBR','NE','NEM','NFLX','NFX','NI','NKE','NKTR','NLY','NOK','NRG','NTNX','NUGT','NVAX','NVDA','NWL','NXPI','NYCB','OAS','OCLR','ODP','OIH','OKE','ON','OPK','ORCL','OSTK','OXY','P','PAA','PAH','PBCT','PBR','PBR.A','PCAR','PCG','PDCO','PE','PEG','PEP','PFE','PG','PHM','PK','PLUG','PM','PPL','PSTG','PTEN','PTI','PYPL','QCOM','QD','QEP','QID','QQQ','RAD','RDS.A','RF','RIG','RIO','ROKU','RRC','RSPP','RSX','S','SAN','SBGL','SBUX','SCHW','SDOW','SDS','SGYP','SH','SIG','SIRI','SLB','SLM','SLV','SM','SMH','SNAP','SO','SPXL','SPXS','SPXU','SPY','SQ','SQQQ','SRC','SRNE','SSO','STNG','STX','SVXY','SWN','SYF','SYMC','SYY','T','TAL','TBT','TEVA','TGT','TJX','TLT','TMUS','TNA','TQQQ','TRQ','TSLA','TSM','TTWO','TV','TVIX','TWTR','TWX','TXN','TZA','UA','UAL','UNH','UNP','UPS','USB','USO','UTX','UVXY','V','VALE','VEA','VEON','VER','VGK','VIAB','VICI','VIPS','VLO','VLY','VNQ','VOD','VOO','VRX','VTI','VTR','VWO','VXX','VZ','WBA','WDC','WEN','WFC','WFT','WIN','WLL','WMB','WMT','WPX','WU','WY','WYNN','X','XBI','XEL','XL','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XOM','XOP','XRT','ZNGA','ZTO']
"""

"""
ETFs ['AAXJ','ACWI','ACWX','AGG','AMJ','AMLP','AMZA','ANGL','ARKK','ASHR','BIL','BIV','BKLN','BND','BNDX','BOTZ','BRZU','BSV','BWX','CIU','COMT','CSJ','CWB','DBA','DBC','DBEF','DBEU','DBJP','DBO','DDM','DGAZ','DGRO','DIA','DOG','DRIP','DUST','DVY','DWT','DXD','DXJ','EDZ','EELV','EEM','EEMO','EEMV','EFA','EFAV','EFG','EFV','EIDO','EMB','EMLC','EMLP','EPI','EPOL','EPP','ERUS','ERX','ERY','EUFN','EUM','EWA','EWC','EWD','EWG','EWH','EWI','EWJ','EWL','EWM','EWP','EWQ','EWS','EWT','EWU','EWW','EWY','EWZ','EZA','EZU','FAS','FAZ','FDN','FEZ','FLOT','FLRN','FNCL','FNDF','FNDX','FPE','FTEC','FTSM','FV','FVD','FXE','FXI','FXY','GASL','GDX','GDXJ','GLD','GOVT','GREK','GSG','GSLC','GSY','GUNR','GUSH','HDGE','HDV','HEDJ','HEFA','HEWG','HEWJ','HEZU','HYD','HYG','HYLB','HYS','IAU','IBB','IDV','IEF','IEFA','IEI','IEMG','IEV','IGF','IJH','IJR','ILF','INDA','INDY','ITA','ITB','ITM','ITOT','IUSG','IUSV','IVE','IVV','IVW','IWB','IWD','IWF','IWM','IWN','IWO','IWP','IWS','IWV','IXUS','IYE','IYF','IYR','IYW','IYZ','JDST','JNK','JNUG','KBE','KBWB','KIE','KRE','KWEB','LABD','LABU','LIT','LQD','MBB','MCHI','MDY','MINT','MLPA','MLPI','MTUM','MUB','NEAR','NOBL','NUGT','OEF','OIH','OIL','PCY','PDBC','PFF','PGX','PHB','PHYS','PSLV','PSQ','QID','QLD','QQQ','QUAL','ROBO','RSP','RSX','RUSL','RWX','SCHA','SCHB','SCHD','SCHE','SCHF','SCHG','SCHH','SCHO','SCHV','SCHX','SCHZ','SCO','SCZ','SDOW','SDS','SDY','SH','SHM','SHV','SHY','SHYG','SJNK','SLV','SMH','SOXL','SOXS','SOXX','SPAB','SPDW','SPEM','SPHD','SPIB','SPLG','SPLV','SPSB','SPSM','SPTM','SPXL','SPXS','SPXU','SPY','SPYG','SPYV','SQQQ','SRLN','SRTY','SSO','SVXY','TBF','TBT','TECL','TFI','TIP','TLT','TMF','TMV','TNA','TQQQ','TVIX','TWM','TZA','UCO','UDOW','UGAZ','UGLD','UNG','UPRO','URA','USLV','USMV','USO','UUP','UVXY','UWT','VB','VBR','VCIT','VCSH','VDE','VEA','VEU','VFH','VGK','VGSH','VGT','VIG','VIXY','VLUE','VMBS','VNM','VNQ','VNQI','VO','VOO','VPL','VT','VTEB','VTI','VTIP','VTV','VUG','VV','VWO','VXUS','VXX','VYM','XBI','XES','XHB','XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XME','XOP','XRT','YINN']
"""


issues = []
issue_output_list = []

# 4-14-18 Future update: only update prices from latest date to current date
data_source = 'morningstar'

start_date = datetime(1990,1,1)
end_date = datetime.today()

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
    print (qt.tail())
    
    issue_name = issue + '.pkl'
    file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
    qt.to_pickle(file_name)
    
    
