'''
Pseudo-spectral solver for the 1D adjoint SWHD equations
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import Adjoint_SWHD_1D

param_path = 'examples/time_marching_adjoint_swhd1D'
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{param_path}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid1D(pm)
solver = Adjoint_SWHD_1D(pm)
NT = ((pm.T/pm.dt) /pm.ostep - 1)
last_time = int(NT*pm.ostep)

# Initial conditions physical fields
uu = np.load(f'{pm.field_path}/uu_{last_time:04}.npy')
hh = np.load(f'{pm.field_path}/hh_{last_time:04}.npy')
hb = np.load(f'{pm.field_path}/hb.npy')

# Measurements
um = np.load(f'{pm.data_path}/uu_{last_time:04}.npy') # this will be um_{t}.npy
hm = np.load(f'{pm.data_path}/hh_{last_time:04}.npy') # hm_{t}.npy

# Null initial conditions for adjoint state
uu_ = np.zeros_like(uu)
hh_ = np.zeros_like(hh)

fields = [uu_, hh_, uu, hh, um, hm, hb]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt(f'{pm.out_path}/balance.dat', unpack=True)
# Plot fields
val = 2*pm.ostep
out_u = np.load(f'{pm.out_path}/adjuu_{val:04}.npy')
out_h = np.load(f'{pm.out_path}/adjhh_{val:04}.npy')
out_hb = np.load(f'{pm.out_path}/hb.npy')

f,axs = plt.subplots(ncols=3)

axs[0].plot(out_hb , label = 'hb')
axs[0].legend()
axs[1].plot(out_u , label = 'u_')
axs[1].legend()
axs[2].plot(out_h , label = 'h_')
axs[2].legend()
plt.savefig(f'{pm.out_path}/fields.png')
