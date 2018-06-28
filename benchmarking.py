# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:59:43 2018

@author: Thomas Schatz

Some benchmarking with random parameters
to get an idea of the size of problems solvable with CPlex.

#TODO: OSF
"""

from docplex.mp.advmodel import AdvModel as Model
import numpy as np
import time

def get_model(u, C, f, c, l):
    m, n = len(c), len(u)
    mdl = Model(name='evophono')
    # create variables (with positivity constraint)
    # is an explicit upper bound for p and q helpful? Seems slightly faster...
    P = mdl.continuous_var_matrix(range(n), range(m), lb=0, ub=1, name='p')
    Q = mdl.continuous_var_matrix(range(m), range(n), lb=0, ub=1, name='q')
    # add summing to one constraints
    for i in range(n):
        mdl.add_constraint(mdl.sum(P[i,j] for j in range(m)) == 1)
    for j in range(m):
        mdl.add_constraint(mdl.sum(Q[j,i] for i in range(n)) == 1)
    # defining objective
    communication_accuracy = mdl.sum(u[i]*P[i,j]*C[j,k]*Q[k,i] 
                                        for i in range(n)
                                            for j in range(m)
                                                for k in range(m))
    production_cost = mdl.sum(f[i]*P[i,j]*c[j]
                                for i in range(n)
                                   for j in range(m))   
    fitness = communication_accuracy - l*production_cost
    # setting objective
    mdl.maximize(fitness)
    return mdl, P, Q


def random_params(m, n, seed=0):
    np.random.seed(seed)
    u = np.random.rand(n)
    C = np.random.rand(m, m)
    C = C / np.tile(np.sum(C, axis=1), (m, 1)).T
    f = np.random.rand(n)
    c = np.random.rand(m)
    l = -1
    return u, C, f, c, l


m = 340 # number of possible signals/words
n = 50 # number of referents/meaning/concepts
params = random_params(m, n)

# 40s: 30 - 150 local; 10 - 20 global?
# 50 - 340 local seems to take at least 1h, perhaps much more...
# perhaps opti will be easier with non random parameters...

"""
# Global
mdl, P, Q = get_model(*params)
# 1 global but need convex, 2 local, 3 global for nonconvex
mdl.parameters.optimalitytarget = 3 
# try to solve the problem
url, key = None, None
t = time.time()
assert mdl.solve(url=url, key=key), "Solve failed"
print(time.time()-t)
mdl.report()

L_global = []
for i in range(n):
    P_i = [P[i,j].solution_value for j in range(m)]
    P_i = np.array([0 if e<10**-5 else e for e in P_i])
    P_i = P_i / sum(P_i)
    assert max(P_i) == 1, P_i
    L_global.append(np.argmax(P_i))
"""

# Local
mdl, P, Q = get_model(*params)
mdl.parameters.optimalitytarget = 2
    
# try to solve the problem
url, key = None, None
t = time.time()
assert mdl.solve(url=url, key=key), "Solve failed"
print(time.time()-t)
mdl.report()

L_local = []
for i in range(n):
    P_i = [P[i,j].solution_value for j in range(m)]
    P_i = np.array([0 if e<10**-5 else e for e in P_i])
    P_i = P_i / sum(P_i)
    assert max(P_i) == 1, P_i
    L_local.append(np.argmax(P_i))

# parallelisation?
# time limit and provisional solution? in local vs global?