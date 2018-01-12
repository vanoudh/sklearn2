# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:31:35 2018
@author: mvanoudh
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier

# Format is
# model_constructor, is_regressor, is_tree, max_try, { parameters }


model_list = [
    (DummyRegressor(), True, False, 10, {
            'mo__strategy': ['mean', 'median']
            }),
    (DummyClassifier(), False, False, 10, {
            'mo__strategy':['stratified', 'most_frequent', 'prior', 'uniform']
            }),

    (LinearRegression(), True, False, 200, {
         'mo__normalize': [True, False]
    }),
    (Ridge(), True, False, 100, {
     'mo__alpha': np.power(10., range(2, -5, -1)),
     'mo__normalize': [True, False]
     }),
    (LogisticRegression(), False, False, 1, {
            }),

    (KNeighborsRegressor(), True, False, 10, {
     'mo__n_neighbors': range(1, 10),
      'mo__algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
     }),
    (KNeighborsClassifier(), False, False, 10, {
         'mo__n_neighbors': range(1, 10),
         'mo__algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
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
         'mo__n_estimators': range(10, 200),
         'mo__max_features': np.arange(.05, .9, step=.1),
         'mo__criterion': ('mae', 'mse'),
         'mo__max_depth': range(1, 10),
         'mo__min_samples_split': range(2, 10),
         'mo__random_state': (0, )
         }),
    (RandomForestClassifier(), False, True, 10, {
         'mo__n_estimators': [50],
         'mo__max_features': [0.6],
         'mo__criterion': ['entropy'],
         'mo__max_depth': [None],
         'mo__min_samples_split': [5],
         'mo__random_state': (0, )
         })
]
