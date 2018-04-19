# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:15:42 2017

@author: mvanoudh
"""

import logging
import numpy as np
import pandas as pd
from numpy.random import seed

from sklearn2.datasets import get_titanic, get_house_prices
from sklearn2.datasets import get_iris, get_boston
from sklearn2.utils import print_summary

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn2.feature_extraction import DateEncoder, SparseCatEncoder
from sklearn2.feature_extraction import ConstantInputer
from sklearn2.decomposition import TruncatedSVD2
from sklearn2.feature_selection import SelectKBest2

#from sklearn.decomposition import TruncatedSVD
from sklearn2.utils import TransformerWrap, PassThrought, todf
from sklearn2.utils import object_cols, numeric_cols


pd.options.display.width = 160

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logging.getLogger().setLevel(level=logging.INFO)

seed(0)



x, y = get_titanic(True)
#x, y = get_iris()

print_summary(x)





ppl = Pipeline([
        ('in', ConstantInputer()),
        ("da", DateEncoder()),
        ('en', FeatureUnion([
               ('nu', Pipeline([('ft', FunctionTransformer()), ("sc", TransformerWrap(StandardScaler()))])),
               ('ca', make_pipeline(FunctionTransformer(), SparseCatEncoder(), FunctionTransformer()))
               ])),
        ('fi', make_union(SelectKBest2(), TruncatedSVD2()))
        ])
    
params = {
        'en__nu__ft__func': lambda x:x[numeric_cols(x)],
        'en__nu__ft__validate': False,
        'en__ca__functiontransformer-1__func': lambda x:x[object_cols(x)],
        'en__ca__functiontransformer-1__validate': False,
        'en__ca__functiontransformer-2__func': lambda x:x.loc[:, x.nunique() > 1],
        'en__ca__functiontransformer-2__validate': False
        }

ppl.set_params(**params)

xt = ppl.fit_transform(x, y)

print_summary(xt)