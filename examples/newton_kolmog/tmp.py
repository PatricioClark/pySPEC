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
from pySPEC.time_marching import KolmogorovFlow
from pySPEC.newton import UPO

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
grid   = ps.Grid2D(pm)
solver = KolmogorovFlow(pm)
newt = UPO(pm, solver)

idx = 0
uu = np.load(f'input/uu_{idx:0{pm.ext}}.npy')
vv = np.load(f'input/vv_{idx:0{pm.ext}}.npy')

fields = [uu, vv]

ffields = [grid.forward(f) for f in fields]

for u, u_ in zip(fields, ffields):
    print(np.linalg.norm(u), np.linalg.norm(u_))

fields_proj = grid.inc_proj(ffields)

for u, u_ in zip(ffields, fields_proj):
    print(np.linalg.norm(u-u_), np.linalg.norm(u))

