# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:55:19 2019

@author: kruegkj
"""
import pandas as pd
import numpy as np
import glob

# print(glob.glob("TLT-Long-System-*/*.csv"))

# get list of sheets in each system dir
sheets = glob.glob("TLT-Long-System-*/*.csv")

all_data = pd.DataFrame()

for i in sheets:
    print(i)
    df = pd.read_csv(i)
    print(df.Features.head(2))
    all_data = all_data.append(df,ignore_index=True)
    
all_data.describe()

