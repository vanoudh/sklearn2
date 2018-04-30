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
from sklearn2.utils import print_summary
from sklearn2.feature_extraction import DateEncoder, SparseCatEncoder
from sklearn2.datasets import get_titanic


pd.options.display.width = 160

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logging.getLogger().setLevel(level=logging.DEBUG)



def test_date_encoder():
    df = get_titanic()
    df.fillna(0, inplace=True)
    df2 = DateEncoder(ascategory=True).fit_transform(df)
    print_summary(df2)


def test_dummy_encoder():
    
    def _test_dummy(x):
        xtr, xte = x[:2], x[2:]
    
        enc = SparseCatEncoder(drop_first=True)
        enc.fit(xtr)
        xtr_e = enc.transform(xtr)
        xtr_e2 = enc.fit_transform(xtr)
        xte_e = enc.transform(xte)
        assert xtr_e.equals(xtr_e2)
        print(xtr_e)
        print(xte_e)

    x = pd.DataFrame({'v1': ['a', 'b', 'b', 'd'],
                      'v2': ['k', 'k', 'm', 'm'],
                       })
    _test_dummy(x)
    _test_dummy(x.values)
      

def test_pipeline():
    x, y = get_titanic(True)

    model = Pipeline([
                    ("da", DateEncoder()), 
                    ("du", SparseCatEncoder()), 
                    ("lr", DummyClassifier())
                    ])
    params = { 
            'da__ascategory': True
            }



    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)

    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


