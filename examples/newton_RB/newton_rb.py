'''
Newton-Hookstep solver for Kolmogorov flow
==========================================
A variable-order RK scheme is used for time integration,
and the 2/3 rule is used for dealiasing.
'''

import numpy as np
import matplotlib.pyplot as plt
import params as pm

import pySPEC as ps
from pySPEC.solvers import SPECTER
from pySPEC.methods import DynSys

pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid2D_semi(pm)
solver = SPECTER(pm)
newt = DynSys(pm, solver)

# Load initial conditions
if pm.restart != 0:
    # Restart Newton Solver from last iteration at index 0 (start of evolution)
    restart_path = f'output/iN{pm.restart:02}/'
    fields = solver.load_fields(restart_path, 0)
    T, sx = newt.get_restart_values(pm.restart) # Get period and shift from last Newton iteration
else:
    # Start Newton Solver from initial guess
    fields = solver.load_fields(pm.input, pm.stat)    
    T, sx = pm.T, pm.sx # Set initial guess for period and shift
    # Create directories
    newt.mkdirs()
    newt.write_header()

U = newt.flatten(fields)
if pm.sx is not None: # If searching for RPOs
    X = np.append(U, [T, sx])
else:
    X = np.append(U, T) # If searching for UPOs with 0 shift

# Iterate Newton Solver
X = newt.run_newton(X)
