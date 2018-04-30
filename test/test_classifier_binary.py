# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:34:26 2017

@author: mvanoudh
"""

#import matplotlib.pyplot as plt  #analysis:ignore
import numpy as np
import pandas as pd
from numpy import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from sklearn2.utils import model_name
from sklearn2.datasets import get_iris

def model_assertivness(clf, x):
    yp = clf.predict_proba(x)
    tmp = np.apply_along_axis(lambda z:(z>0.98).sum(), 1, yp)
    return tmp.mean()
    
random.seed(0)

pd.options.display.width = 160

x, y = get_iris(True)
y = (y == 'virginica')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


rfc = RandomForestClassifier(n_estimators=100, oob_score=True, class_weight='balanced', random_state=0)
dtc = DecisionTreeClassifier(criterion='entropy', max_features=0.85,
                             min_samples_split=2,
                             random_state=0, max_depth=14)


def test_run():
    for m in [rfc, dtc]:
        print(model_name(m))
        m.fit(x_train, y_train)
        print('score in and out of sample', m.score(x_train, y_train), m.score(x_test, y_test))
        y_proba = m.predict_proba(x_test)[:, 1]
        print(roc_auc_score(y_test, y_proba))
    
    
