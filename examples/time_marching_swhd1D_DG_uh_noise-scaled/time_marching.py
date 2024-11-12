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
from noise import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# Initialize solver
grid   = ps.Grid1D(pm)

# create noisy u, h
uu,hh = uh_noise(pm, grid)

# create noisey hb
s0 =  0.1
s1 =  0.3
s2 = 1.4
s3 = 0.05
s4 = 0.2
s5 = 0.8
hb = s0*np.exp(-(grid.xx-np.pi/s2)**2/s1**2) + s3*np.exp(-(grid.xx-np.pi/s5)**2/s4**2)
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

f,axs = plt.subplots(ncols=3, figsize = (15,3))

axs[0].plot(out_hb , label = '$h_b$')
axs[0].legend()
axs[1].plot(out_u , label = '$u$')
axs[1].legend()
axs[2].plot(out_h , label = '$h$')
axs[2].legend()
plt.savefig(f'{pm.out_path}/fields.png')
