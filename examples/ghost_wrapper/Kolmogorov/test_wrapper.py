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
grid = ps.Grid2D(pm)
solver = GHOST(pm)
lyap = DynSys(pm, solver)

# Load initial velocity field

# path = 'input'
# fields = solver.load_fields(path, 0)

uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)

fields = [uu, vv]

# Plot initial fields
for field, ftype in zip(fields, solver.ftypes):
    plt.figure()
    plt.imshow(field, cmap='viridis')
    plt.colorbar()
    plt.title(ftype)
    plt.show()
    plt.savefig(f'{ftype}_i.png')

T = 1.0
# If you only need last fields
fields = solver.evolve(fields, T)

# If you need intermediate fields
# fields = solver.evolve(fields, T, bstep=pm.bstep, ostep=pm.ostep, sstep=pm.sstep, opath=pm.opath, bpath=pm.bpath,spath=pm.spath)

# Plot final fields
for field, ftype in zip(fields, solver.ftypes):
    plt.figure()
    plt.imshow(field, cmap='viridis')
    plt.colorbar()
    plt.title(ftype)
    plt.show()
    plt.savefig(f'{ftype}_f.png')


# # Calculate n Lyapunov exponents
# n = 50
# eigval_H, eigvec_H, Q = newt.lyap_exp(fields, T, n, tol = 1e-10)

# # Save Lyapunov exponents
# spath = f'lyap/'
# os.makedirs(spath, exist_ok=True)
# np.save(f'{spath}lyap_exp.npy', eigval_H)
# np.save(f'{spath}eigvec_H.npy', eigvec_H)
# np.save(f'{spath}Q.npy', Q)
