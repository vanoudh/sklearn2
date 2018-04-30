# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:10:21 2017

@author: mvanoudh
"""

#import numpy as np
import pandas as pd

from sklearn2.utils import split_xy, print_summary
from sklearn2.datasets import get_titanic_clean
from sklearn2.covariance import compute_correls

pd.options.display.width = 160

df = get_titanic_clean()['Fare Age Sex Embarked Pclass Survived'.split()]

print_summary(df)


def test1():
    return compute_correls(df)

    
def test2():
    return compute_correls(df, model='entropy')

    
def test3():
    return compute_correls(df, model='linear')


def test4():
    return compute_correls(df, model='forest')

"""     
if __name__ == "__main__":
    res1 = test1()
    res2 = test2()
    res3 = test3()
    res4 = test4()
 """