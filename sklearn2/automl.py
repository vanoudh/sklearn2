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

from category_encoders import OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, check_cv
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.scorer import roc_auc_scorer, r2_scorer
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import f_regression, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from sklearn2.feature_extraction import DateEncoder, SparseCatEncoder
from sklearn2.feature_extraction import ConstantInputer, NumericFilter
from sklearn2.decomposition import TruncatedSVD2
from sklearn2.utils import TransformerWrap, PassThrought, numeric_cols, object_cols
from sklearn2.utils import model_name
from sklearn2.feature_selection import SelectKBest2
from sklearn2.feature_selection import f_forest_regression, f_linear_regression
from sklearn2.feature_selection import f_forest_classification, f_linear_classification
from sklearn2.metrics import avg_roc_auc_scorer
from sklearn2.automl_models import model_list

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


def get_selector(is_regressor, is_tree):
    if is_regressor:
        return [f_regression, mutual_info_regression]
    else:
        return [f_classif, mutual_info_classif]


def get_pipeline(est, is_tree, is_regressor, params):
    name = model_name(est)
    if name.startswith('Dummy'):
        ppl = Pipeline([
                       ('ft', FunctionTransformer()), 
                       ('mo', est)
                      ])
        params['ft__func'] = [lambda x:x[numeric_cols(x)]]
        params['ft__validate'] = [False]
    elif is_tree:
        ppl = Pipeline([
                       ('da', DateEncoder()),
                       ('du', OrdinalEncoder()),
                       ('ft', FunctionTransformer()),
                       ('se', SelectKBest2()),
                       ('mo', est)
                      ])
        params['da__ascategory'] = [False]
        params['du__drop_invariant'] = [True]
        params['ft__func'] = [lambda x:x.fillna(-999)]
        params['ft__validate'] = [False]
        params['se__score_func'] = get_selector(is_regressor, is_tree)
        params['se__k'] = [0.2, 0.5, 0.8, 1000, 1000]
    else:
        ppl = Pipeline([
                ('da', DateEncoder()),
                ('en', FeatureUnion([
                       ('nu', Pipeline([('ft', FunctionTransformer()), ('in', Imputer()), ('sc', TransformerWrap(StandardScaler()))])),
                       ('ca', Pipeline([('ft', FunctionTransformer()), ('sc', SparseCatEncoder())]))
                       ])),
                ('fu', FeatureUnion([('se', SelectKBest2()), ('dr', TruncatedSVD2())])),
                ('mo', est)
                ])
            
        params['en__nu__ft__func'] = [lambda x:x[numeric_cols(x)]]
        params['en__nu__ft__validate'] = [False]
        params['en__ca__ft__func'] = [lambda x:x[object_cols(x)]]
        params['en__ca__ft__validate'] = [False]
        params['fu__se__score_func'] = get_selector(is_regressor, is_tree)
        params['fu__se__k'] = [0.2, 0.5, 0.8, 1000]
        params['fu__dr__k'] = [0.2, 0.5, 0.8, 1000]        
        
    return name, ppl, params


def get_pipelines(x, y):    
    is_regressor, is_binary = get_type(y)               
    ppl_list = []
    for est, is_reg, is_tree, n_iter, params in model_list:
        if is_reg != is_regressor:
            continue
        name, ppl, params = get_pipeline(est, is_tree, is_regressor, params)
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
    

def get_cv(cv, y):
    if not isinstance(cv, str):
        return cv
    sep = ':'
    if cv.find(sep) < 0:
        cv += sep + '3'
    cvtype, n = cv.split(sep)
    if cvtype != 'ts':
        raise ValueError('unexpected type:' + cvtype)
    return TimeSeriesSplit(n_splits=int(n))

    
def get_search_models(x, y, scorer, cv, iter_factor, verbose):
    ppl_list = get_pipelines(x, y)
    cv = get_cv(cv, y)
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
    best_sm = None
    sms = get_search_models(x, y, scorer, cv, iter_factor, verbose) 
    for name, sm in sms:
        sm.fit(x, y)
        if best_sm == None or sm.best_score_ > best_sm.best_score_:
            best_sm = sm
        print('{:>24} test score : {:.4f}'.format(name, sm.best_score_))
    return best_sm

