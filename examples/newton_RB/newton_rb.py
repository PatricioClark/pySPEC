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
from pySPEC.time_marching import SPECTER
from pySPEC.linalg import UPO

pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid2D_semi(pm)
solver = SPECTER(pm)
newt = UPO(pm, solver)

# Load initial conditions
X = newt.load_ic()

# Iterate Newton Solver
X = newt.iterate(X)
