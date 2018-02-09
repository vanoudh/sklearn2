# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:31:35 2018
@author: mvanoudh
"""

import numpy as np
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

# Format is
# model_constructor, is_regressor, is_tree, max_try, { parameters }


model_list = [
    (DummyRegressor(), True, False, 10, {
        'mo__strategy': ['mean', 'median']
    }),
    (DummyClassifier(), False, False, 10, {
        'mo__strategy':['stratified', 'most_frequent', 'prior', 'uniform']
    }),

    (Ridge(), True, False, 100, {
        'mo__alpha': np.power(10., range(2, -5, -1)),
        'mo__normalize': [True, False]
     }),
    (LogisticRegression(), False, False, 1, {
        'mo__penalty': ['l1', 'l2'],
        'mo__C': [1.0, 0.1, 10]
    }),

    (KNeighborsRegressor(), True, False, 10, {
        'mo__n_neighbors': range(1, 10),
        'mo__algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
     }),
    (KNeighborsClassifier(), False, False, 10, {
        'mo__n_neighbors': range(1, 10),
        'mo__algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute'),
     }),

    (DecisionTreeRegressor(), True, True, 1, {
        'mo__max_features': ('auto', 'sqrt', 'log2'),
        'mo__criterion': ('mae', 'mse',),
        'mo__max_depth': range(1, 5),
        'mo__min_impurity_decrease': np.arange(0., 10e-7, step=1e-7),
        'mo__min_samples_split': range(2, 5),
        'mo__min_samples_leaf': range(1, 10),
        'mo__random_state': (0, )     
     }),
    (DecisionTreeClassifier(), False, True, 10, {
        'mo__max_features': np.arange(.05, 1, step=.05),
        'mo__criterion': ('entropy',),
        'mo__max_depth': range(3, 20),
        'mo__min_impurity_decrease': np.arange(1e-7, 10e-7, step=1e-7),
        'mo__min_samples_split': range(2, 5),
        'mo__random_state': (0, )     
     }),

    (RandomForestRegressor(), True, True, 10, {
        'mo__n_estimators': range(10, 100),
        'mo__max_features': np.arange(.05, .9, step=.1),
        'mo__criterion': ('mae', 'mse'),
        'mo__max_depth': [0.2, 0.5, 0.8, None],
        'mo__min_samples_split': range(2, 10),
        'mo__random_state': (0, )
    }),
    (RandomForestClassifier(), False, True, 10, {
        'mo__n_estimators': range(10, 100),
        'mo__max_features': np.arange(.05, .9, step=.1),
        'mo__criterion': ['entropy', 'gini'],
        'mo__max_depth': [0.2, 0.5, 0.8, None],
        'mo__min_samples_split': range(2, 10),
        'mo__random_state': (0, )
    }),

    (ExtraTreesRegressor(), True, True, 10, {
         'mo__n_estimators': range(10, 100),
         'mo__max_features': np.arange(.05, .9, step=.1),
         'mo__criterion': ('mae', 'mse'),
         'mo__max_depth': [0.2, 0.5, 0.8, None],
         'mo__min_samples_split': range(2, 10),
         'mo__random_state': (0, )
    }),
    (ExtraTreesClassifier(), False, True, 10, {
         'mo__n_estimators': range(10, 100),
         'mo__max_features': np.arange(.05, .9, step=.1),
         'mo__criterion': ['entropy', 'gini'],
         'mo__max_depth': [0.2, 0.5, 0.8, None],
         'mo__min_samples_split': range(2, 10),
         'mo__random_state': (0, )
    }),

    (GradientBoostingRegressor(), True, True, 10, {
         'mo__loss':['ls', 'lad', 'huber'],
         'mo__learning_rate':[0.1, 0.5],
         'mo__n_estimators': range(10, 100),
         'mo__max_depth': [3, 5, 10, None],
         'mo__max_features': np.arange(.05, .9, step=.1),
         'mo__criterion': ('mae', 'friedman_mse'),
         'mo__min_samples_split': [2, 5],
         'mo__random_state': (0, )
    }),
    (GradientBoostingClassifier(), False, True, 10, {
         'mo__loss':['deviance'],
         'mo__learning_rate':[0.1, 0.5],
         'mo__n_estimators': range(10, 100),
         'mo__max_depth': [3, 5, 10, None],
         'mo__max_features': np.arange(.05, .9, step=.1),
         'mo__criterion': ('mae', 'friedman_mse'),
         'mo__min_samples_split': [2, 5],
         'mo__random_state': (0, )
    })
]
