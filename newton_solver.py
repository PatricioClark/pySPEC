import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from gmres_kol import GMRES

import params2D as pm
import mod2D as mod

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)
apply_A = mod.application_function(evolve, grid, pm)

### Feed initial aproximation extracted with Kolmogorov solver
path = '/share/data4/jcullen/pySPEC/run1/'
t_i = 135
dt = 1e-3
f_name = f'fields_{int(t_i/dt):06}'
fields = np.load(f'{path}output/{f_name}.npz')
uu = fields['uu']
vv = fields['vv']
X = mod.flatten_fields(uu, vv)
T = 100

# Initial guess
Y  = evolve(X, T)
dt = 0

# Apply A to initial vector
Ax = apply_A(X, Y , T, dX=X-Y, dT=0)

# TODO: modificar GMRES y arnoldi_step para que usen apply_A
# TODO: usar numba jit
