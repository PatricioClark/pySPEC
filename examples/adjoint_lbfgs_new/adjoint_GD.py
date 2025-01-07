import json
import numpy as np
import matplotlib.pyplot as plt
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrd
import optax

from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

from mod import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
fpm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
fpm.Lx = 2*np.pi*fpm.Lx
fpm.out_path = fpm.forward_out_path
check_dir(fpm.out_path) # make out path if it doesn't exist
fpm.ostep = fpm.forward_ostep

# Parse JSON into an object with attributes corresponding to dict keys.
bpm = json.load(open(f'{current_dir}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
bpm.Lx = 2*np.pi*bpm.Lx
bpm.out_path = bpm.backward_out_path
check_dir(bpm.out_path) # make out path if it doesn't exist
bpm.ostep = bpm.backward_ostep

# Initialize grid
grid   = ps.Grid1D(fpm)
fsolver = SWHD_1D(fpm)
bsolver = Adjoint_SWHD_1D(bpm, fsolver)
# get measurments for backpropagation
bsolver.get_measurements()

# total number of iterations
total_iterations = bpm.iitN - bpm.iit0 - 1

# remove all files from hb_path if restarting GD
check_dir(bpm.hb_path)

# restart from iit0
reset(fpm, bpm)

# Initial conditions
v1 = 0.00025
v2 =  0.5
v3 = 2
uu0 = v1 * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2)

c1 = 5e-5
c2 = 0.5
c3 = 2
hh0 = fpm.h0 + c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2)

# true hb
fsolver.update_true_hb()
bsolver.update_true_hb()


# Initial hb ansatz
hb = np.zeros_like(fsolver.true_hb)
dg = np.zeros_like(fsolver.true_hb)
# update hb and dg
fsolver.update_hb(hb)
fsolver.update_dg(dg)

# choose gradient descent optimizer
if pm.optimizer == 'lbfgs':
    # Define objective
    dim = 1024
    mat = jrd.normal(jrd.PRNGKey(0), (dim, dim))
    mat = mat @ mat.T  # Ensure mat is positive semi-definite
    # Define optimizer
    lr = bpm.lgd
    opt = optax.scale_by_lbfgs()
    # Initialize optimization
    state = opt.init(hb)
elif pm.optimizer == 'sgd':
    # momentum gradient descent optimizer
    # Initialize the momentum optimizer
    opt_init, opt_update = optax.sgd(learning_rate=bpm.lgd, momentum=0.9)
    # Gradient update step
    def update(params, grad, opt_state):
        updates, opt_state = opt_update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    opt_state = opt_init(hb)




for iit in range(fpm.iit0 + 1, fpm.iitN):
    # update iit
    fpm.iit = iit
    bpm.iit = iit

    # catch initial conditions for foward integration
    uu = uu0
    hh = hh0
    fields = [uu, hh]

    # Forward integration
    print(f'iit {iit} : evolving forward')
    fsolver.evolve(fields, fpm.T, bstep=fpm.bstep, ostep=fpm.ostep)
    bsolver.update_fields(fsolver)

    # Null initial conditions for adjoint state
    uu_ = np.zeros_like(grid.xx)
    hh_ = np.zeros_like(grid.xx)
    fields = [uu_, hh_]
    # update backward solver at each step
    bsolver.get_sparse_forcing()


    # Backward integration
    print(f'\niit {iit} : evolving backward')
    fields = bsolver.evolve(fields, bpm.T, bstep=bpm.bstep, ostep=bpm.ostep)

    # integrate h_*ux from T to t=0
    print(f'\niit {iit} : calculate dg/dhb')
    Nt = round(fpm.T/fpm.dt)
    dg = np.trapz( bsolver.hx_uu, dx = 1e-4, axis = 0)
    # dg = np.trapz( np.load(f'{bpm.out_path}/hx_uu_memmap.npy'), dx = 1e-4, axis = 0)
    # update hb values with momentum GD
    print(f'\niit {iit} : update hb')
    # Run optimization
    if pm.optimizer == 'lbfgs':
        DG, state = opt.update(dg, state, hb)
        hb = hb - lr * DG
    elif pm.optimizer == 'sgd':
        hb, opt_state = update(hb, dg, opt_state)

    # update hb and dg
    fsolver.update_hb(hb)
    fsolver.update_dg(dg)

    # save for the following iteration
    print(f'\niit {iit} : save hb')
    bsolver.update_loss(iit-1)
    bsolver.update_val(iit-1)

    # save loss
    # loss = [f'{iit}', f'{u_loss:.6e}' , f'{h_loss:.6e}']
    # with open(f'{fpm.hb_path}/loss.dat', 'a') as output:
    #     print(*loss, file=output)

    # calculate validation
    # save validation
    # val = [f'{iit}', f'{hb_val:.6e}']
    # with open(f'{fpm.hb_path}/hb_val.dat', 'a') as output:
    #     print(*val, file=output)

    if iit%pm.ckpt==0:
        # Plot fields
        print(f'\niit {iit} : plot')
        tval = int(fpm.T/fpm.dt*0.5)
        out_u = bsolver.uus[tval]
        true_u = bsolver.uums[tval]
        out_h = bsolver.hhs[tval]
        true_h = bsolver.hhms[tval]

        plt.close("all")
        plot_fields(fpm,
                    hb,
                    fsolver.true_hb,
                    out_u,
                    true_u,
                    out_h,
                    true_h)
        plot_fourier(fpm, grid, hb,
                    fsolver.true_hb)

        # plot loss and val
        # loss = np.loadtxt(f'{fpm.hb_path}/loss.dat', unpack=True)
        # val = np.loadtxt(f'{fpm.hb_path}/hb_val.dat', unpack=True)
        np.save(f'{fpm.hb_path}/u_loss.npy', bsolver.u_loss)
        np.save(f'{fpm.hb_path}/h_loss.npy', bsolver.h_loss)
        np.save(f'{fpm.hb_path}/validation.npy', bsolver.val)
        plot_loss(fpm, bsolver.u_loss, bsolver.h_loss, bsolver.val)
        plot_dg(fpm,dg,DG)
        print(f'done iit {fpm.iit}')
