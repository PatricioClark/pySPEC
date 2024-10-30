'''
Pseudo-spectral solver for the 1D SWHD equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D

param_path = 'examples/time_marching_swhd1D_3'
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{param_path}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid1D(pm)
solver = SWHD_1D(pm)

# Initial conditions
v1 = 0.05
v2 =  0.3927
v3 = 2
v4 = 0.001
noise = np.zeros_like(grid.xx)
for ki in range(10,20+1):
    noise = noise + v4*np.cos(ki*grid.xx)
uu = v1 * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2) + noise

c1 = 0.05
c2 = 0.3927
c3 = 2
hh = pm.h0 + c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2) + noise

s0 =  0.5
s1 = 0.3
s3 = 1
s4 = 0.001
noise_hb = np.zeros_like(grid.xx)
for ki in range(10,100+1):
    noise_hb = noise_hb + s4*np.cos(ki*grid.xx)

hb = s0*np.exp(-(grid.xx-np.pi)**2/s1**2) + noise_hb # the true hb
np.save(f'{pm.hb_path}/hb_{pm.iit}.npy', hb)
fields = [uu, hh]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt(f'{pm.out_path}/balance.dat', unpack=True)
# Plot fields
val = 2*pm.ostep
out_u = np.load(f'{pm.out_path}/uu_memmap.npy', mmap_mode='r')[pm.ostep*1000] # all uu fields in time
out_h = np.load(f'{pm.out_path}/hh_memmap.npy', mmap_mode='r')[pm.ostep*1000] # all uu fields in time
out_hb = np.load(f'{pm.out_path}/hb.npy')

f,axs = plt.subplots(ncols=3)

axs[0].plot(out_hb , label = 'hb')
axs[0].legend()
axs[1].plot(out_u , label = 'u')
axs[1].legend()
axs[2].plot(out_h , label = 'h')
axs[2].legend()
plt.savefig(f'{pm.out_path}/fields.png')
