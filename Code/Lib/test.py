# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:33:58 2018

@author: KRUEGKJ


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model, model_selection, preprocessing, svm
from sklearn.metrics import mean_squared_error, r2_score
from yellowbrick.regressor import PredictionError, ResidualsPlot
"""

import sys
before = [str(m) for m in sys.modules]
import my_module
after = [str(m) for m in sys.modules]
print [m for m in after if not m in before]