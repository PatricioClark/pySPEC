'''
GHOST wrapper example for Kolmogorov Flow
'''

import numpy as np
import matplotlib.pyplot as plt
import os

import pySPEC as ps
from pySPEC.solvers import GHOST
from pySPEC.methods import DynSys
import params as pm

pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
grid = ps.Grid3D(pm)
solver = GHOST(pm, solver='ROTH', dimension=3)
newt = DynSys(pm, solver)

# Load initial conditions
if pm.restart_iN == 0:
    fields = solver.load_fields(pm.input, pm.stat)
    sx = pm.sx # Set initial guess for shift
    # Create directories
    newt.mkdirs()
    newt.write_header()
else:
    # Restart Newton Solver from last iteration at index 0 (start of evolution)
    restart_path = f'output/iN{pm.restart_iN:02}/'
    fields = solver.load_fields(restart_path, 0)
    _, sx = newt.get_restart_values(pm.restart_iN) # Get shift from last Newton iteration

U = newt.flatten(fields)
X = newt.form_X(U, sx=sx)

# Iterate Newton Solver
X = newt.run_newton(X)