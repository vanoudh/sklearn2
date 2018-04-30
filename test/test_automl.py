# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:15:42 2017

@author: mvanoudh
"""

import logging
import numpy as np
import pandas as pd
import time
from numpy.random import seed

from sklearn2.datasets import get_titanic, get_house_prices, get_iris, get_boston
from sklearn2.automl import get_best_model

pd.options.display.width = 160

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logging.getLogger().setLevel(level=logging.INFO)

seed(0)


def print_sm(sm):
    print('Pipeline:')
    for s in sm.best_estimator_.steps:
        print('  ', s[1])


def test_run():
    for data_func in [get_boston]:
        print('dataset:', str(data_func).split()[1])
        x, y = data_func(True)    
        m = get_best_model(x, y, None, None, 1) 
        print_sm(m)
    
