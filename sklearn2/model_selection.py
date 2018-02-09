# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:41:32 2017

@author: mvanoudh
"""

import numpy as np
import pandas as pd
import sklearn
#import sklearn.cross_validation
from sklearn.base import clone


def cross_val_score2(estimator, X, y=None, scoring=None, 
                     parameters=None, cv=None, verbose=0, fit_params=None, 
                     early_stop=False, stopping_threshold=None,
                     return_last_estimator=False,
                     return_predict = False,
                     return_predict_proba = False
                    ):
    """Evaluate a score by cross-validation

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        
    parameters : dict or None
        Parameters to be set on the estimator.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    early_stop : boolean
        if True the cross_validation stop if the result are bad. 
    
    Stops if : NaN in the score, or after at least 3 iteration if the best score is bellow the 'stopping_threshold'
        
    stopping_threshold : float
        cross_validation stops if best score is bellow (after 3 iterations) if early_stop = True

    return_last_estimator : boolean (default = False)
        if True, the last estimator is also return (to test)
        
    return_predict : boolean (default = False)
        if True, the cross validated prediction will be returned as well
        
    return_predict_proba : boolean (default = False)
        if True, the cross validated prediction of proba will be returned as well

    Returns
    -------
    scores : pd.DataFrame, shape = len(cv) , 4.
    
    The columns are :
    train : score on training data (in-sample score)
    test  : score on testing data (out-of-sample score)
    test_size : the number of observation in the testing set
    time : the time it took to fit
    
    if return_last_estimator is True, the scores and the last estimator are returned


    """
    #Rmk : the function is mostly a copy of sklearn cross_val_score with a few additions    
    
    
    if early_stop and stopping_threshold is None:
        raise ValueError("I need a stopping_threshold when 'early_stop' is True")
    
    X, y = sklearn.cross_validation.indexable(X, y)

    cv = sklearn.cross_validation.check_cv(cv, X, y, classifier=sklearn.cross_validation.is_classifier(estimator))
    scorer = sklearn.cross_validation.check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    all_scores = list()
    i = 1

    if return_predict:
        y_hat = np.empty(y.shape,dtype=y.dtype)
    
    if return_predict_proba:
        y_hat_proba = []
        
    for train, test in cv:
        if verbose > 0:
            print("cv %d started\n" % i )
            
        new_estimator = clone(estimator)
        
        temp_res = sklearn.cross_validation._fit_and_score(new_estimator,X,y, scorer, train, test, verbose, parameters, fit_params,return_train_score=True)
        score_train, score_test, nb_of_observations, time_of_fit = temp_res
        
        ### New ###
        # TODO : faire un truc predict OU predict_proba
        if return_predict:
            y_hat[test] = new_estimator.predict(X[test,:])
            
        if return_predict_proba:
            df_proba = pd.DataFrame(new_estimator.predict_proba(X[test,:]),columns  = list(new_estimator.classes_) , index = test)
            y_hat_proba.append(df_proba)
#            pr_proba = new_estimator.predict_proba(X[test,:])
#            if y_hat_proba is None:
#                classes = new_estimator.classes_
#                y_hat_proba = np.empty((len(y),len(classes)))
#                
#            assert np.all(new_estimator.classes_ == classes)
#            y_hat_proba[test,:] = pr_proba
        
        if verbose >0:
            print("cv %d done!\n\n" % i)
        if verbose > 1:
            print("score train : %2.2f%% , score test : %2.2f%%" % (100*score_train,100*score_test))
            
        all_scores.append((score_train, score_test,
                           nb_of_observations, time_of_fit))
        
        if early_stop:
            if pd.isnull(score_test) or pd.isnull(score_train):
                if verbose:
                    print("I'll stop cross validation now (NaNs)")
                break
            
        if early_stop and i >= 3:
            all_score_train,all_score_test,__,__ = zip(*all_scores)
            if np.max(all_score_test) <= stopping_threshold:
                if verbose:
                    print("I'll stop cross validation now (bad perfs)")
                break
            
        i += 1
            
    result = pd.DataFrame(all_scores,
                          columns=["train", "test", "test_size", "time"])
    
    if return_predict_proba:            
        y_hat_proba = pd.concat(y_hat_proba ,axis=0, 
                                ignore_index=False).sort_index()
        for c in y_hat_proba.columns:
            y_hat_proba.loc[y_hat_proba[c].isnull(), c] = 0.0
        
    if return_last_estimator:
        if return_predict:
            return result, new_estimator, y_hat
        elif return_predict_proba:
            return result, new_estimator, y_hat_proba        
        else:
            return result, new_estimator
    else:
        if return_predict:
            return result, y_hat
        elif return_predict_proba:
            return result, y_hat_proba
        else:
            return result
        
        

