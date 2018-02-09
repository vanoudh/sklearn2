# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:41:32 2017

@author: mvanoudh
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
   
    
def labels_to_binary(y, classes, dtype=np.int8):
    yb = np.zeros((len(y), len(classes)), dtype)
    for i, cl in enumerate(classes):
        yb[:, i] = y == cl
    return yb
    
def _accuracy_at_proba(y_true, y_proba, proba):
    sure_proba = (y_proba > proba) + 0
    sure_samples = sure_proba.sum(axis=1) >= 1    
    y_true_sub = np.argmax(y_true[sure_samples], axis=1)
    y_pred_sub = np.argmax(y_proba[sure_samples], axis=1)    
    return accuracy_score(y_true_sub, y_pred_sub)

def _coverage_at_proba(y_proba, proba):
    sure_proba = (y_proba > proba) + 0
    sure_samples = sure_proba.sum(axis=1) >= 1    
    return sure_samples.mean()    

# scorers

class accuracy_at_proba(object):
    def __init__(self, proba):
        self.proba = proba
    def __call__(self, clf, X, y, sample_weight=None):        
        y_true = labels_to_binary(y, clf.classes_)
        y_proba = clf.predict_proba(X)                  
        return _accuracy_at_proba(y_true, y_proba, self.proba)
        
class coverage_at_proba(object):
    def __init__(self, proba):
        self.proba = proba
    def __call__(self, clf, X, y, sample_weight=None):        
        y_proba = clf.predict_proba(X)                  
        return _coverage_at_proba(y_proba, self.proba)
            
class avg_roc_auc_scorer(object):
    def __init__(self, average="macro"):
        self.average = average        
        self._deprecation_msg = None
    def __call__(self, clf, X, y, sample_weight=None):        
        y_true = labels_to_binary(y, clf.classes_)
        y_proba = clf.predict_proba(X)                  
        return roc_auc_score(y_true, y_proba, sample_weight=sample_weight, 
                             average=self.average)

class log_loss_scorer2(object):
    def __init__(self):
        self._deprecation_msg = None
    def __call__(self, clf, X, y, sample_weight=None):
        y_proba = clf.predict_proba(X)
        return -1.0 * log_loss(y, y_proba, sample_weight=sample_weight,
                               labels=clf.classes_)
  
class confidence_score(object):
    """ Mesure how much 'maxproba' helps discriminate between mistaken and correct instance 
    If the maximum probability is high, the model is confident in its prediction otherwise the model esitates.
    We'd like error to be less present when maximum proba is high.
    
    roc_auc_score mesures how much 'maxproba' discriminates betweeen mistaken and correct instance
    
    """ 
    def __init__(self):
        self._deprecation_msg = None
        
    def __call__(self, clf, X, y, sample_weight=None):

        yhat = clf.predict(X)
        yhat_proba = clf.predict_proba(X)
        
        if isinstance(yhat_proba, pd.DataFrame):
            yhat_proba = yhat_proba.values
            
        yhat_maxproba = yhat_proba.max(axis=1)
        
        is_correct = 1*(yhat == y)
        if (is_correct == 1).all():
            return np.nan
        else:
            return roc_auc_score(y_true=is_correct, 
                                 y_score=yhat_maxproba,
                                 sample_weight=sample_weight)

            
sklearn.metrics.scorer.SCORERS["avg_roc_auc"] = avg_roc_auc_scorer() 
sklearn.metrics.scorer.SCORERS["avg_roc_auc_micro"] = avg_roc_auc_scorer(average = 'micro') 
sklearn.metrics.scorer.SCORERS["avg_roc_auc_macro"] = avg_roc_auc_scorer(average = 'macro') 
sklearn.metrics.scorer.SCORERS["log_loss2"] = log_loss_scorer2() 




        
#
#y_true = np.array([[0, 1],
#                   [1, 0],
#                   [1, 0]])
#                  
#y_proba = np.array([[.25, .75],
#                    [.95, .05],
#                    [.40, .60]])
#
#for level in [.2, 0.4, 0.6, 0.8, 0.9]:
#    print(level, _score_at_proba(y_true, y_proba, level))