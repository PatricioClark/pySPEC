'''
Pseudo-spectral solver for the 1D Kuramoto-Sivashinsky equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D

param_path = 'examples/time_marching_swhd1D'
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{param_path}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Initialize solver
grid   = ps.Grid1D(pm)
solver = SWHD_1D(pm)

# Initial conditions
uu = (0.05*np.cos(2*np.pi*3.0*grid.xx/pm.Lx)
      )
hh = pm.h0 + (0.05*np.cos(2*np.pi*3.0*grid.xx/pm.Lx)
      )
fields = [uu, hh]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt(f'{pm.out_path}/balance.dat', unpack=True)
# Plot fields
val = 2*pm.ostep
out_u = np.load(f'{pm.out_path}/uu_{val:04}.npy')
out_h = np.load(f'{pm.out_path}/hh_{val:04}.npy')

f,axs = plt.subplots(ncols=3)

axs[0].plot(bal[0], bal[1] , label = 'balance')
axs[0].legend()
axs[1].plot(out_u , label = 'u')
axs[1].legend()
axs[2].plot(out_h , label = 'h')
axs[2].legend()
plt.savefig(f'{pm.out_path}/fields.png')
