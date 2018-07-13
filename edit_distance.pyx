# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:25:28 2018

@author: Thomas Schatz

Implementation in Cython of the classic Wagner-Fischer algorithm for computing
edit distances (cf. https://en.wikipedia.org/wiki/Edit_distance).
"""

import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t CTYPE_t # distance type
CTYPE = np.float64 # also distance type 
ctypedef np.intp_t IND_t # array index type


def edit_distance(a, b, s_cost, d_cost, i_cost):
    # Do input checks?
    #   s, d, i size should be consistent
    #   a, b should contain values consistent with this size (O-indexed)
    #   maybe also d and i costs should be larger than max of s costs?
    return _edit_distance(a, b, len(a), len(b), s_cost, d_cost, i_cost)


cpdef _edit_distance(IND_t[:] a, IND_t[:] b, IND_t len_a, IND_t len_b,
                     CTYPE_t[:,:] s_cost, CTYPE_t[:] d_cost, CTYPE_t[:] i_cost):
    # initialisation
    cdef IND_t i, j
    cdef CTYPE_t[:,:] dis = np.empty((1+len_b, 1+len_a), dtype=CTYPE)
    dis[0,0] = 0
    for i in range(1, 1+len_b):
        dis[i,0] = dis[i-1,0] + d_cost[b[i-1]] 
    for j in range(1, 1+len_a):
        dis[0,j] = dis[0,j-1] + i_cost[a[j-1]]
    # main part
    for i in range(1, 1+len_b):
        for j in range(1, 1+len_a):
            if b[i-1] == a[j-1]:
                dis[i,j] = dis[i-1,j-1]
            else:
                dis[i,j] = min(dis[i-1,j] + d_cost[b[i-1]],
                               dis[i,j-1] + i_cost[a[j-1]],
                               dis[i-1,j-1] + s_cost[a[j-1], b[i-1]])
    return dis[len_b, len_a]
