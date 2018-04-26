# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:36:21 2017
@author: mvanoudh
"""

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

'''
    gradient bandit
'''

k = 10
ne = 100
ns = 1000
alpha1 = '1/n'   # step size or '1/n'
alpha2 = 0.1

def one(a):
    z = np.zeros(k)
    z[a] = 1
    return z

def action(p):
    a = np.random.multinomial(1, p)
    return a.argmax()

def bandit(a):
    return rr[a] + np.random.normal()

def alpha(alpha0, n):
    return 1/n if alpha0 == '1/n' else alpha0
    
score = np.zeros((ne, ns))

for experiment in range(ne):
    q = 0
    h = np.zeros(k)
    rr = np.random.normal(size=k)
    cum_reward = 0
    for step in range(ns):
        exph = np.exp(h)
        p = exph / exph.sum()
        a = action(p)
        r = bandit(a)
        q += alpha(alpha1, step + 1) * (r - q)
        h += alpha(alpha2, step + 1) * (r - q) * (one(a) - p)  
        score[experiment, step] = r
    
plt.plot(score.mean(axis=0))
print(score.mean(), score.std())