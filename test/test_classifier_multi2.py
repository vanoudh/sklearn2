# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:34:26 2017

@author: mvanoudh
"""

import pandas as pd
from numpy import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from sklearn2.utils import model_name
from sklearn2.datasets import get_iris

# here we use the multi output format for multiclass classif
# this make it possible to use roc_auc_score
    
random.seed(0)

pd.options.display.width = 160

x, y = get_iris(True)

y2 = pd.get_dummies(y).values

x_train, x_test, y_train, y_test, y2_train, y2_test = train_test_split(x, y, y2, test_size=0.33, random_state=42)


rfc = RandomForestClassifier(n_estimators=100, oob_score=True, class_weight='balanced', random_state=0)
dtc = DecisionTreeClassifier(criterion='entropy', max_features=0.85,
                             min_samples_split=2,
                             random_state=0, max_depth=14)
logit = LogisticRegression()


def test_run():
    for m in [rfc, dtc, logit]:
        print(model_name(m))
        m.fit(x_train, y_train)
        print(m.classes_)
        print('score in and out of sample', m.score(x_train, y_train), m.score(x_test, y_test))
        y_proba = m.predict_proba(x_test)
        print(roc_auc_score(y2_test, y_proba, None))
    
    
