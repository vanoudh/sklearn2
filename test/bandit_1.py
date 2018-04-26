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
    epsilon greedy action    
'''

k = 10
ne = 100
ns = 1000
q0 = 0          # optimistic initial value ( 5 is good )
eps = 0         # epsilon greedy parameter  ( 0.1 is good )
alpha = '1/n'   # step size or '1/n'
c = 0          # upper confidence bound param ( 1 is good )


def action(q, n):
    t = n.sum() + 1
    epsilon_true = np.random.randint(0, 100)/100 < eps
    q_ucb = q + c * np.sqrt(np.log(t)/(n + 1))
    explo_a = np.random.randint(0, k)
    return explo_a if epsilon_true else q_ucb.argmax()


def bandit(a):
    return rr[a] + np.random.normal()


score = np.zeros((ne, ns))


for experiment in range(ne):    
    q = np.zeros(k) + q0            # initial values of the action-value
    n = np.zeros(k) 
    rr = np.random.normal(size=k)   # real rewards
    cum_reward = 0
    for step in range(ns):
        a = action(q, n)
        r = bandit(a)
        n[a] += 1
        alpha_step = 1/n[a] if alpha == '1/n' else alpha
        q[a] += alpha_step * (r - q[a])
        score[experiment, step] = r
    
plt.plot(score.mean(axis=0))
print(score.mean(), score.std())