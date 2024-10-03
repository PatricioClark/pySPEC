'''
Pseudo-spectral solver for the 1D SWHD equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D

param_path = 'adjoint/make_data_swhd1D'
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{param_path}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid1D(pm)
solver = SWHD_1D(pm)

# Initial conditions
v1 = 0.05
v2 = 0.3927
v3 = 2
uu = v1 * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2)

c1 = 0.05
c2 = 0.3927
c3 = 2
hh = pm.h0 + c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2)

s1 = 0.5
s2 = 0.3
s3 = 1
hb = s1*np.exp(-(grid.xx-np.pi/s3)**2/s2**2)

fields = [uu, hh, hb]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt(f'{pm.out_path}/balance.dat', unpack=True)
# Plot fields
val = 2*pm.ostep
out_u = np.load(f'{pm.out_path}/uu_{val:04}.npy')
out_h = np.load(f'{pm.out_path}/hh_{val:04}.npy')
out_hb = np.load(f'{pm.out_path}/hb.npy')

f,axs = plt.subplots(ncols=3)

axs[0].plot(out_hb , label = 'hb')
axs[0].legend()
axs[1].plot(out_u , label = 'u')
axs[1].legend()
axs[2].plot(out_h , label = 'h')
axs[2].legend()
plt.savefig(f'{pm.out_path}/fields.png')
