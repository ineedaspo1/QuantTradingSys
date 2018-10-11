# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:20:40 2018

@author: kruegkj
"""

if __name__ == "__main__":
    from retrieve_data import *
    from ta_momentum_studies import *
    from ta_volume_studies import *
    from ta_volatility_studies import *
    from ta_overlap_studies import *
    from transformers import *
    from oscillator_studies import *
    from candle_indicators import *
    taLibVolSt = TALibVolumeStudies()
    taLibMomSt = TALibMomentumStudies()
    transf = Transformers()
    oscSt = OscialltorStudies()
    vStud = TALibVolatilityStudies()
    feat_gen = FeatureGenerator()
    candle_ind = CandleIndicators()
    taLibVolSt = TALibVolumeStudies()
    custVolSt = CustVolumeStudies()
    taLibOS = TALibOverlapStudies()
    functionDict = {
                "RSI"               : taLibMomSt.RSI,
                "PPO"               : taLibMomSt.PPO,
                "CMO"               : taLibMomSt.CMO,
                "CCI"               : taLibMomSt.CCI,
                "ROC"               : taLibMomSt.rate_OfChg,
                "UltimateOscillator": taLibMomSt.UltOsc,
                "Normalized"        : transf.normalizer,
                "Zscore"            : transf.zScore,
                "Scaler"            : transf.scaler,
                "Center"            : transf.centering,
                "Lag"               : transf.add_lag,
                "DetrendPO"         : oscSt.detrend_PO,
                "ATR"               : vStud.ATR,
                "NATR"              : vStud.NATR,
                "ATRRatio"          : vStud.ATR_Ratio,
                "DeltaATRRatio"     : vStud.delta_ATR_Ratio,
                "BBWidth"           : vStud.BBWidth,
                "HigherClose"       : candle_ind.higher_close,
                "LowerClose"        : candle_ind.lower_close,
                "ChaikinAD"         : taLibVolSt.ChaikinAD,
                "ChaikinADOSC"      : taLibVolSt.ChaikinADOSC,
                "OBV"               : taLibVolSt.OBV,
                "MFI"               : taLibVolSt.MFI,
                "ease_OfMvmnt"      : custVolSt.ease_OfMvmnt,
                "exp_MA"            : taLibOS.exp_MA,
                "simple_MA"         : taLibOS.simple_MA,
                "weighted_MA"       : taLibOS.weighted_MA,
                "triple_EMA"        : taLibOS.triple_EMA,
                "triangMA"          : taLibOS.triangMA,
                "dblEMA"            : taLibOS.dblEMA,
                "kaufman_AMA"       : taLibOS.kaufman_AMA,
                "delta_MESA_AMA"    : taLibOS.delta_MESA_AMA,
                "inst_Trendline"    : taLibOS.inst_Trendline,
                "mid_point"         : taLibOS.mid_point,
                "mid_price"         : taLibOS.mid_price,
                "pSAR"              : taLibOS.pSAR
                }
            
    dSet = DataRetrieve()
    dSet.save_obj(functionDict, 'func_dict')
    funcDict = dSet.load_obj('func_dict')
    print(funcDict)