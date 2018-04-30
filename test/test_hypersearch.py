# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:34:26 2017

@author: mvanoudh
"""

#import matplotlib.pyplot as plt  #analysis:ignore
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from numpy import random

from sklearn2.datasets import get_iris
from sklearn2.utils import model_name

random.seed(0)

pd.options.display.width = 160

x, y = get_iris(True)
 

models = [
    (DummyClassifier(), 1, {}),
    (KNeighborsClassifier(), 30, {
     'n_neighbors': range(1, 10),
      'algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
     }),
    (DecisionTreeClassifier(), 10, {
         'max_features': np.arange(.05, 1, step=.05),
         'criterion': ('entropy',),
         'max_depth': range(3, 20),
         'min_impurity_decrease': np.arange(0., 10e-7, step=1e-7),
         'min_samples_split': range(2, 5),
         'random_state': (0, )     
     }),
    (RandomForestClassifier(), 1, {
         'n_estimators': range(20, 200),
         'max_features': np.arange(.05, .9, step=.1),
         'criterion': ('gini', 'entropy'),
         'max_depth': range(1, 15),
         'min_samples_split': range(2, 10),
         'random_state': (0, )
         })
]

cvs = ShuffleSplit(10)

def test_run():
    for m in models:
        reg, n_iter, params = m
        gs = RandomizedSearchCV(reg, param_distributions=params, cv=cvs.split(x), 
                                verbose=1, n_iter=n_iter)
        gs.fit(x, y)
        print(model_name(reg), gs.best_score_)
 