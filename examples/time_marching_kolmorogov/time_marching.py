'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC.pseudo as ps
from pySPEC.time_marching import KolmogorovFlow

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L
pm.nu = 1.0/40.0

# Initialize solver
grid   = ps.Grid2D(pm)
solver = KolmogorovFlow(pm)

# Initial conditions
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
fu = grid.forward(uu)
fv = grid.forward(vv)
fu, fv = grid.inc_proj([fu, fv])
uu = grid.inverse(fu)
vv = grid.inverse(fv)
fields = [uu, vv]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep)

# Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)
plt.plot(bal[0], bal[1])

# Plot fields
uu, vv = fields
u2 = uu**2 + vv**2
plt.figure()
plt.imshow(u2)
plt.show()
