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

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.scorer import roc_auc_scorer, r2_scorer
#from sklearn.cross_validation import check_cv

from sklearn2.utils import print_summary
from sklearn2.utils import model_name, split_xy
from sklearn2.feature_extraction import DateEncoder, DummyEncoder
from sklearn2.feature_selection import SelectKBest2
from sklearn2.feature_selection import f_forest_regression, f_linear_regression
from sklearn2.feature_selection import f_forest_classification, f_linear_classification
from sklearn2.datasets import get_titanic, get_boston, get_iris
from sklearn2.decomposition import TruncatedSVD2
from sklearn2.metrics import avg_roc_auc_scorer

#from dstk.data_encoder import MyNumericalEncoder
#from dstk.data_extractor import MyFeaturesSelector
#from dstk.data_toolbox import my_cross_val_score_V2, my_shuffle

seed(0)

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logging.getLogger().setLevel(level=logging.INFO)

pd.options.display.width = 160

#x, y = get_titanic()    # binary target
#x, y = get_boston()     # continuous target
#x, y = get_iris()       # 3 class target

#def get_deals():
#    path = r'K:\casestudies\deals\adeals_h2o.csv'
#    df = pd.read_csv(path)
#    return split_xy(df, 's')

#for data_func in get_boston, get_titanic, get_iris:
for data_func in [get_titanic]:    
    x, y = data_func()
#    x, y = my_shuffle(x, y, seed=0)
#    print_summary(x)
    
    tot = type_of_target(y)
    is_regressor = tot in ['continuous', 'continuous-multioutput']
    is_binary = tot == 'binary'
    
    print('The target is {}'.format(tot))
    
    regressor_models = [
        (DummyRegressor(), 10, {}),
        (LinearRegression(), 200, {
             'mo__normalize': [True, False]
        }),
        (Ridge(), 100, {
         'mo__alpha': np.power(10., range(2, -5, -1)),
         'mo__normalize': [True, False]
         }),
        (KNeighborsRegressor(), 10, {
         'mo__n_neighbors': range(1, 10),
          'mo__algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
         }),
        (DecisionTreeRegressor(), 1, {
             'mo__max_features': ('auto', 'sqrt', 'log2'),
             'mo__criterion': ('mae', 'mse',),
             'mo__max_depth': range(1, 5),
             'mo__min_impurity_decrease': np.arange(0., 10e-7, step=1e-7),
             'mo__min_samples_split': range(2, 5),
             'mo__min_samples_leaf': range(1, 10),
             'mo__random_state': (0, )     
         }),
        (RandomForestRegressor(), 10, {
             'mo__n_estimators': range(10, 200),
             'mo__max_features': np.arange(.05, .9, step=.1),
             'mo__criterion': ('mae', 'mse'),
             'mo__max_depth': range(1, 10),
             'mo__min_samples_split': range(2, 10),
             'mo__random_state': (0, )
             })
    ]
    
    classifier_models = [
        (DummyClassifier(), 1, {}),
#        (KNeighborsClassifier(), 10, {
#             'mo__n_neighbors': range(1, 10),
#             'mo__algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
#         }),
#        (DecisionTreeClassifier(), 10, {
#             'mo__max_features': np.arange(.05, 1, step=.05),
#             'mo__criterion': ('entropy',),
#             'mo__max_depth': range(3, 20),
#             'mo__min_impurity_decrease': np.arange(1e-7, 10e-7, step=1e-7),
#             'mo__min_samples_split': range(2, 5),
#             'mo__random_state': (0, )     
#         }),
        (LogisticRegression(), 1, {}),
        (RandomForestClassifier(), 10, {
             'mo__n_estimators': [50],
             'mo__max_features': [0.6],
             'mo__criterion': ['entropy'],
             'mo__max_depth': [10],
             'mo__min_samples_split': [5],
             'mo__random_state': (0, )
             })
    ]
    
    
    res = {}
    _verbose = 1
    _cv = 3 #check_cv(3, x, y, ~is_regressor)
    
    if is_regressor:
        models = regressor_models
        selection_scorers = [f_linear_regression, f_forest_regression]
        scorer = r2_scorer
        greater_is_better = True
    else:
        models = classifier_models
        selection_scorers = [f_linear_classification, f_forest_classification]
        scorer = roc_auc_scorer if is_binary else avg_roc_auc_scorer()
        greater_is_better = True
        
    sign = 1 if greater_is_better else -1
    
    for m in models:
        est, n_iter, params = m
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
    
        grid_size = len(ParameterGrid(params))
        if grid_size < n_iter:
            gs = GridSearchCV(ppl, params, scorer, cv=_cv, verbose=_verbose)
        else:
            gs = RandomizedSearchCV(ppl, params, n_iter, scorer, cv=_cv, 
                                    verbose=_verbose, random_state=0)
        gs.fit(x, y)
        res[model_name(est)] = gs.best_params_
        ppl.set_params(**gs.best_params_)
        ppl.fit(x, y)
        print('{:>24} scores train and test : {:.4f} {:.4f}'.format(name, 
              sign*scorer(ppl, x, y), sign*gs.best_score_))
    
 

