'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import json
import importlib
from types import SimpleNamespace

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Import corresponding module
mod     = importlib.import_module(f'mods.mod_ps{pm.dim}D', package=None)
sol_mod = importlib.import_module(f'solvers.{pm.solver}', package=None)

# Initialize solver
grid   = mod.Grid(pm)
solver = getattr(sol_mod, pm.solver)(grid, pm)

# Initial conditions
fields = mod.initial_conditions(grid, pm)

# Evolve
fields = solver.evolve(fields, pm.T)
