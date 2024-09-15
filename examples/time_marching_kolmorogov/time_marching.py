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
pm = json.load(open('params_time_marching_kolm.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 1.0
pm.Ly = 1.0
pm.nu = 1.0/40.0

# Import corresponding module
import pseudo as ps
from solvers import KolmogorovFlow

# Initialize solver
grid   = ps.Grid2D(pm)
solver = KolmogorovFlow(pm)

# Initial conditions
fields = ps.initial_conditions2D(grid, pm)

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep)

# Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)
plt.plot(bal[0], bal[1])

# Plot fields
fu, fv = fields
uu = ps.inverse(fu)
vv = ps.inverse(fv)
u2 = uu**2 + vv**2
plt.figure()
plt.imshow(u2)
plt.show()
