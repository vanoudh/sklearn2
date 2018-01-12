# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:26:06 2018
@author: mvanoudh
"""

import logging
import numpy as np
import pandas as pd
import time
from numpy.random import seed

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, check_cv
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.scorer import roc_auc_scorer, r2_scorer
#from sklearn.cross_validation import check_cv

from sklearn2.utils import model_name
from sklearn2.feature_extraction import DateEncoder, DummyEncoder
from sklearn2.feature_selection import SelectKBest2
from sklearn2.feature_selection import f_forest_regression, f_linear_regression
from sklearn2.feature_selection import f_forest_classification, f_linear_classification
from sklearn2.decomposition import TruncatedSVD2
from sklearn2.metrics import avg_roc_auc_scorer
from sklearn2.automl_models import model_list
#from dstk.data_encoder import MyNumericalEncoder
#from dstk.data_extractor import MyFeaturesSelector
#from dstk.data_toolbox import my_cross_val_score_V2, my_shuffle

seed(0)

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logging.getLogger().setLevel(level=logging.INFO)

pd.options.display.width = 160


def get_type(y):
    tot = type_of_target(y)
    if y.dtype != 'object' and len(np.unique(y)) > 100:
        tot = 'continuous'
    is_regressor = tot in ['continuous', 'continuous-multioutput']
    is_binary = tot == 'binary'
    return is_regressor, is_binary

    
def get_pipelines(x, y):    
    
    is_regressor, is_binary = get_type(y)               
    if is_regressor:
        selection_scorers = [f_linear_regression, f_forest_regression]
    else:
        selection_scorers = [f_linear_classification, f_forest_classification]        
    
    ppl_list = []
    for m in model_list:
        if m[1] == is_regressor:
            est, _, _, n_iter, params = m
            name = model_name(est)
            ppl = Pipeline([
                           ("da", DateEncoder()), 
                           ("du", DummyEncoder()),
                           ("dr", TruncatedSVD2()),
                           ("se", SelectKBest2()), 
                           ("mo", est)
                          ])
            params['da__ascategory'] = [True]
            params['du__drop_first'] = [False]
            params['dr__k'] = [1000]
            params['se__k'] = [1000]
            params['se__score_func'] = selection_scorers        
            ppl_list.append((name, ppl, params, n_iter))

    return ppl_list


def get_search_model(ppl, params, scorer, cv, n_iter, verbose):
    grid_size = len(ParameterGrid(params))
    if grid_size < n_iter:
        gs = GridSearchCV(ppl, params, scorer, cv=cv, verbose=verbose)
    else:
        gs = RandomizedSearchCV(ppl, params, n_iter, scorer, cv=cv, 
                                verbose=verbose, random_state=0)
    return gs
    

def cv_factory(cv, y):
    return TimeSeriesSplit() if cv == 'ts' else cv

    
def get_search_models(x, y, scorer, cv, iter_factor, verbose):
    ppl_list = get_pipelines(x, y)
    cv = cv_factory(cv, y)
    print('scorer:', scorer)
    print('cv:', cv)
    for name, ppl, params, n_iter in ppl_list:
        yield name, get_search_model(ppl, 
                                     params, 
                                     scorer, 
                                     cv, 
                                     n_iter*iter_factor, 
                                     verbose)


def get_best_model(x, y, scorer=None, cv=None, iter_factor=1, verbose=0):
    best_sm, best_score = None, -np.inf
    sms = get_search_models(x, y, scorer, cv, iter_factor, verbose) 
    for name, sm in sms:
        sm.fit(x, y)
        bs = sm.best_score_
        if  bs > best_score:
            best_sm, best_score = sm, bs
        print('{:>24} test score : {:.4f}'.format(name, bs))
    return best_sm.best_estimator_

