'''
Newton-Hookstep solver for Kolmogorov flow
==========================================
A variable-order RK scheme is used for time integration,
and the 2/3 rule is used for dealiasing.
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SPECTER
from pySPEC.newton import UPO

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# Load Newton parameters
pmN = json.load(open('params_newton.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Initialize solver
grid   = ps.Grid2D_wrap(pm)
solver = SPECTER(pm)
newt = UPO(pm, pmN, solver)

# Load initial conditions
fields, sx, T = newt.load_ic()

# Iterate Newton Solver
X = newt.iterate(fields, sx, T)
