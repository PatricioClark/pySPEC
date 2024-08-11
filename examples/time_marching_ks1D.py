'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import json

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params_time_marching_ks1D.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Import corresponding module
import pySPEC.pseudo as ps
from pySPEC.solvers import KuramotoSivashinsky

# Initialize solver
grid   = ps.Grid1D(pm)
solver = KuramotoSivashinsky(pm)

# Initial conditions
fields = ps.initial_conditions1D(grid, pm)

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=10000)

# Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)
plt.plot(bal[0], bal[1])

# Plot fields
acc = []
for ostep in range(0,int(100*pm.T/pm.dt)):
    if ostep > 99999:
        break
    out = np.load(f'uu_{ostep*100:04}.npy')
    acc.append(out)

acc = np.array(acc)
plt.figure()
plt.imshow(acc, extent=[0,22,0,100])
plt.show()
