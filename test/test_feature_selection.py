# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:10:21 2017

@author: mvanoudh
"""

import logging
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn2.feature_extraction import DateEncoder, SparseCatEncoder
from sklearn2.feature_selection import RfAutoSelector
from sklearn2.datasets import get_titanic_clean

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logging.getLogger().setLevel(level=logging.DEBUG)

pd.options.display.width = 160


def test_run():
    x, y = get_titanic_clean(True)
    model = Pipeline([
                    ("da", DateEncoder()), 
                    ("du", SparseCatEncoder()), 
                    ("rf", RfAutoSelector()), 
                    ("lr", DummyClassifier())
                    ])
    params = { 
            'da__ascategory': True
            }

    model.set_params(**params)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)

    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
