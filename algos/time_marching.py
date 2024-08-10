'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import json
from types import SimpleNamespace

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Import corresponding module
import pySPEC.pseudo as ps
import solvers

# Initialize solver
grid   = getattr(ps, f'Grid{pm.dim}D')(pm)
solver = getattr(solvers, pm.solver)(grid, pm)

# Initial conditions
fields = ps.initial_conditions(grid, pm)

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep)
