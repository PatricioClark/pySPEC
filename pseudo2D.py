'''
Pseudo-spectral solver for the 2D Navier-Stokes equations

A variable-order RK scheme is used for time integrationa and the 2/3 rule is
used for dealiasing.

See Eq. (6.143) in Pope's Turbulent flows for details.

Kolmogorov forcing.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

import params2D as pm
import mod2D as mod

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)
apply_A = mod.application_function(evolve, grid, pm)

# Initial conditions
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
fu = mod.forward(uu)
fv = mod.forward(vv)
fu, fv = mod.inc_proj(fu, fv, grid)
uu = mod.inverse(fu)
vv = mod.inverse(fv)
X = mod.flatten_fields(uu, vv)

# Run
Xt = evolve(X, 1.0)

plt.figure()
uu, vv = mod.unflatten_fields(X, pm)
plt.imshow(uu)
plt.show()
