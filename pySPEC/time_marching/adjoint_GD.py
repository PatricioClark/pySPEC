''' 1D Adjoint Shallow Water Equations '''
import json
import numpy as np
import matplotlib.pyplot as plt
import os

import jax.numpy as jnp
import jax.random as jrd
import optax

from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

from mod import *

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class Adjoint_GD(PseudoSpectral):
    ''' Performs Gradient Descent optimization for SWHD inverse problems.
    '''

    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)
        self.memmap = pm.memmap # True if saving to disk
        self.iit = pm.iit
        # self.sample_rate = pm.sample_rate
        self.dt_step = pm.dt_step
        self.dx_step = pm.dx_step
        self.random_sample = pm.random_sample
        self.total_steps =  round(self.pm.T/self.pm.dt)
        self.data_path = pm.data_path
        self.field_path = pm.field_path
        self.hb_path = pm.hb_path



# Define objective
# w_opt = hb_true
dim = 1024
mat = jrd.normal(jrd.PRNGKey(0), (dim, dim))
mat = mat @ mat.T  # Ensure mat is positive semi-definite

# Define optimizer
lr = bpm.lgd
opt = optax.scale_by_lbfgs()

# Initialize optimization
state = opt.init(hb)

for iit in range(fpm.iit0 + 1, fpm.iitN):
    # update iit
    fpm.iit = iit
    bpm.iit = iit

    # initialize forward solver at each step
    fsolver = SWHD_1D(fpm)
    # catch initial conditions for foward integration
    uu = uu0
    hh = hh0
    fields = [uu, hh]
    # Forward integration
    print(f'iit {iit} : evolving forward')
    fsolver.evolve(fields, fpm.T, bstep=fpm.bstep, ostep=fpm.ostep)

    # Null initial conditions for adjoint state
    uu_ = np.zeros_like(grid.xx)
    hh_ = np.zeros_like(grid.xx)
    fields = [uu_, hh_]
    # initialize backward solver at each step
    bsolver = Adjoint_SWHD_1D(bpm)
    # Backward integration
    print(f'\niit {iit} : evolving backward')
    fields = bsolver.evolve(fields, bpm.T, bstep=bpm.bstep, ostep=bpm.ostep)

    # integrate h_*ux from T to t=0
    print(f'\niit {iit} : calculate dg/dhb')
    Nt = round(fpm.T/fpm.dt)
    dg = np.trapz( np.load(f'{bpm.out_path}/hx_uu_memmap.npy', mmap_mode='r'), dx = 1e-4, axis = 0)

    # update hb values with momentum GD
    print(f'\niit {iit} : update hb')
    # Run optimization
    u, state = opt.update(dg, state, hb)
    hb = hb - lr * u

    # save for the following iteration
    print(f'\niit {iit} : save hb')
    # Save hb and dg using memmap after each iteration
    save_memmap(f'{fpm.hb_path}/hb_memmap.npy', hb, iit,  total_iterations)
    save_memmap(f'{fpm.hb_path}/dg_memmap.npy', dg, iit,  total_iterations)

    # calculate loss for new fields
    uus =  np.load(f'{bpm.field_path}/uu_memmap.npy', mmap_mode='r') # all uu fields in time
    hhs =  np.load(f'{bpm.field_path}/hh_memmap.npy', mmap_mode='r') # all hh fields in time
    uums =  np.load(f'{bpm.data_path}/uu_memmap.npy', mmap_mode='r')[:Nt] # all uu measurements in time
    hhms=  np.load(f'{bpm.data_path}/hh_memmap.npy', mmap_mode='r')[:Nt] # all hh measurements in time
    u_loss = np.sum((uums - uus)**2)
    h_loss = np.sum((uums - uus)**2)

    loss = [f'{iit}', f'{u_loss:.6e}' , f'{h_loss:.6e}']
    with open(f'{fpm.hb_path}/loss.dat', 'a') as output:
        print(*loss, file=output)

    hb_val = np.sum((true_hb - hb)**2)
    loss = [f'{iit}', f'{hb_val:.6e}']
    with open(f'{fpm.hb_path}/hb_val.dat', 'a') as output:
        print(*loss, file=output)

    # Plot fields
    print(f'\niit {iit} : plot')
    tval = int(fpm.T/fpm.dt*0.5)
    out_u = np.load(f'{fpm.out_path}/uu_memmap.npy', mmap_mode='r')[tval] # all uu fields in time
    out_h = np.load(f'{fpm.out_path}/hh_memmap.npy', mmap_mode='r')[tval] # all uu fields in time

    plt.close("all")
    plot_fields(fpm,hb,true_hb,out_u,out_h)
    plot_fourier(fpm, grid, hb,
                true_hb)


    loss = np.loadtxt(f'{fpm.hb_path}/loss.dat', unpack=True)
    plot_loss(fpm, loss)

    print(f'done iit {fpm.iit}')
