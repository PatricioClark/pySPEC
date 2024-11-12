'''
Pseudo-spectral solver for the 1D SWHD equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import os
import pySPEC as ps
from pySPEC.time_marching import SWHD_1D
from noise import hb_noise

current_dir = os.path.dirname(os.path.abspath(__file__))
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid1D(pm)

# Initial conditions
v1 = 0.00025
v2 =  0.5
v3 = 2
uu = v1 * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2)

c1 = 5e-5
c2 = 0.5
c3 = 2
hh = pm.h0 + c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2)

# create noisey hb
hb = hb_noise(pm = pm, grid= grid, ks =  [20, 50 , 70, 100 , 200, 500], A = 0.001)
np.save(f'{pm.hb_path}/hb_{pm.iit}.npy', hb)
np.save(f'{pm.hb_path}/outs/hb.npy', hb) # the fixed, true hb for adjoint loop later
fields = [uu, hh]

# Evolve
solver = SWHD_1D(pm)
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt(f'{pm.out_path}/balance.dat', unpack=True)
# Plot fields
val = 2*pm.ostep
out_u = np.load(f'{pm.out_path}/uu_memmap.npy', mmap_mode='r')[pm.ostep*1000] # all uu fields in time
# out_h = np.load(f'{pm.out_path}/hh_{val:04}.npy')
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
