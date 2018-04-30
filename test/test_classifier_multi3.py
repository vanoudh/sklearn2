# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:34:26 2017

@author: mvanoudh
"""

import pandas as pd
from numpy import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn2.utils import model_name
from sklearn2.datasets import get_iris
from sklearn2.metrics import accuracy_at_proba, coverage_at_proba

random.seed(0)

pd.options.display.width = 160

x, y = get_iris(True)

cv = StratifiedKFold(3)

rfc = RandomForestClassifier(random_state=0)
dtc = DecisionTreeClassifier()
logit = LogisticRegression()

def test_run():
    for m in [rfc, dtc, logit]:
        print(model_name(m))
        print(cross_val_score(m, x, y, scoring = "accuracy", cv = cv))
        print(cross_val_score(m, x, y, scoring = accuracy_at_proba(1/3), cv = cv))
        print(cross_val_score(m, x, y, scoring = coverage_at_proba(1/3), cv = cv))
        print(cross_val_score(m, x, y, scoring = "avg_roc_auc_macro", cv = cv))
        print(cross_val_score(m, x, y, scoring = "avg_roc_auc_micro", cv = cv))
    
    
