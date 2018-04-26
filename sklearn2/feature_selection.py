# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:09:57 2016

@author: mvanoudh
"""

import matplotlib
matplotlib.use('agg')

import logging
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest

from sklearn2.utils import ratio2int, get_forest, forest_sort


logger = logging.getLogger(__name__)

 
def f_forest_regression(X, y):
    """ return features importances for classification problems based on RandomForest """
    forest = RandomForestRegressor(n_estimators=100)
    forest.fit(X,y)
    return forest.feature_importances_
    
    
def f_forest_classification(X, y):
    """ return features importances for regression problems based on RandomForest """
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X,y)
    return forest.feature_importances_
    
    
def f_linear_regression(X, y):
    """ return features importances for regression problems based on RidgeRegression """
    scaler = StandardScaler(with_mean=False)
    ridge = Ridge()
    ridge.fit(scaler.fit_transform(X), y)
    if ridge.coef_.ndim == 1:
        return np.abs(ridge.coef_)
    else:
        return np.sum(np.abs(ridge.coef_), axis=0)
    
    
def f_linear_classification(X,y):
    """ return features importances for classification problems based on LogisticRegression """
    scaler = StandardScaler(with_mean=False)
    logit = LogisticRegression()
    logit.fit(scaler.fit_transform(X), y)   
    if logit.coef_.ndim == 1:
        return np.abs(logit.coef_)
    else:
        return np.sum(np.abs(logit.coef_), axis=0)
        
        
class SelectKBest2(BaseEstimator, TransformerMixin):
    
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k
    
    def fit(self, X, y):
        p = X.shape[1]
        k_int = ratio2int(p, self.k)
        self.donothing = (k_int <= 0 or k_int >= p)
        if self.donothing:
            return self
        selector = SelectKBest(score_func=self.score_func, k=k_int)
        selector.fit(X, y)
        self.support = selector.get_support()
        return self
        
    def transform(self, X):       
        if self.donothing:
            return X
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support]  # ok for DataFrame
        else:            
            return X[:, self.support]       # ok for np.array and scipy sparse
      
    
class FeatureFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, nfirst=1000, expr='.*'):
        self.nfirst = nfirst
        self.expr = expr

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            Z = X.ix[:, 0:self.nfirst]
            match = re.compile(self.expr).match
            tokeep = [c for c in Z.columns if match(c)]
            return Z[tokeep].as_matrix()
        else:
            return X[:, 0:self.nfirst]
            

class RfAutoSelector(BaseEstimator, TransformerMixin):
    """
    Select columns with RF predictive power
    """    
    def __init__(self):
        self.columns = None
    
    def _good_score(self):
        nburn = len(self.scores) // 10
        return max(self.scores) - np.std(self.scores[nburn:])
        
    def fit(self, X, y=None):       
        logging.info('fit {}'.format(X.shape))
        xs = forest_sort(X, y)
        ff = FeatureFilter()
        rf = get_forest(y)()
        model = Pipeline([("ff", ff), ("rf", rf)])
        params = { 
                   'ff__nfirst': 1, 
                   'rf__n_estimators': 20,
                   'rf__oob_score':  True,
                   'rf__max_features': 'sqrt',
                   'rf__criterion': 'entropy',
                   'rf__random_state': 0
                   }
        
        def oobscore(n):
            params['ff__nfirst'] = n
            model.set_params(**params)
            model.fit(xs, y)
            return model.steps[1][1].oob_score_
        
        self.nr = range(1, xs.shape[1])
        self.scores = list(map(oobscore, self.nr))        
        good_score = self._good_score()
        nf = np.where(self.scores > good_score)[0][0] + 1
        logger.debug('n features {}'.format(nf))
        self.columns = xs.columns[:nf]
        return self

    def transform(self, X, y=None):
        logger.debug('transform {}'.format(X.shape))
        return X[self.columns]

    def plot(self):
        plt.scatter(self.nr, self.scores)
        good_score = self._good_score()
        plt.plot([min(self.nr), max(self.nr)], [good_score, good_score])
        
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()