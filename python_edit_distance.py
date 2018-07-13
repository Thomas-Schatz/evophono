# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:08:39 2018

@author: Thomas Schatz

Pure python version of the edit_distance function to test and debug
cython version.
"""
import numpy as np
import edit_distance as ED


def edit_distance(a, b, s_cost, d_cost, i_cost):
    # initialisation
    dis = np.empty((1+len(b), 1+len(a)))
    dis[0,0] = 0
    for i in range(1, 1+len(b)):
        dis[i,0] = dis[i-1,0] + d_cost[b[i-1]] 
    for j in range(1, 1+len(a)):
        dis[0,j] = dis[0,j-1] + i_cost[a[j-1]]
    # main part
    for i in range(1, 1+len(b)):
        for j in range(1, 1+len(a)):
            if b[i-1] == a[j-1]:
                dis[i,j] = dis[i-1,j-1]
            else:
                dis[i,j] = min(dis[i-1,j] + d_cost[b[i-1]],
                               dis[i,j-1] + i_cost[a[j-1]],
                               dis[i-1,j-1] + s_cost[a[j-1], b[i-1]])
    return dis[len(b), len(a)]


def test_ED(a, b):
    # a, b must be lower case alphabetic strings
    insertion_costs = np.ones(26)
    deletion_costs = np.ones(26)
    substitution_costs = 0.5*np.ones((26, 26))
    a = np.array([ord(e)-97 for e in a], dtype=np.int)
    b = np.array([ord(e)-97 for e in b], dtype=np.int)
    dis = edit_distance(a, b,
                        substitution_costs, deletion_costs, insertion_costs)
    dis2 = ED.edit_distance(a, b, substitution_costs, deletion_costs, insertion_costs)
    return dis, dis2