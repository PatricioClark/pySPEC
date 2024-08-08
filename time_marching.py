'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import numpy as np

from types import SimpleNamespace
import importlib
import json

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Import corresponding module
mod = importlib.import_module(f'mod_ps{pm.dim}D', package=None)
solvers = importlib.import_module(f'solvers{pm.dim}D', package=None)

# Initialize solver
grid   = mod.Grid(pm)
evolve = getattr(solvers, pm.solver)(grid, pm)

# Initial conditions
fields = mod.initial_conditions(grid, pm)

# Evolve
for step in range(int(pm.T/pm.dt)):
    fields = evolve(fields, pm.dt)

    if step % pm.save_freq == 0:
        mod.balance(grid, fields)

