'''
Pseudo-spectral solver for the 1D Kuramoto-Sivashinsky equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import swhd_1d

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Initialize solver
grid   = ps.Grid1D(pm)
solver = swhd_1d(pm)

# Initial conditions
uu = (0.3*np.cos(2*np.pi*3.0*grid.xx/pm.Lx) +
      0.4*np.cos(2*np.pi*5.0*grid.xx/pm.Lx) +
      0.5*np.cos(2*np.pi*4.0*grid.xx/pm.Lx) 
      )
fields = [uu, hh]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)
plt.plot(bal[0], bal[1])

# Plot fields
acc_u = []
acc_h = []
for ii in range(0,int(pm.T/pm.dt), pm.ostep):
    out = np.load(f'uu_{ii:04}.npy')
    acc_u.append(out)
    out = np.load(f'hh_{ii:04}.npy')
    acc_h.append(out)

acc_u = np.array(acc_u)
plt.figure()
plt.imshow(acc_u, extent=[0,22,0,100])
plt.show()

acc_h = np.array(acc_h)
plt.figure()
plt.imshow(acc_h, extent=[0,22,0,100])
plt.show()
