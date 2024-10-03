'''
Pseudo-spectral solver for the 1D SWHD equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

param_path = 'adjoint'
# Parse JSON into an object with attributes corresponding to dict keys.
fpm = json.load(open(f'{param_path}/forward_params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
fpm.Lx = 2*np.pi*fpm.Lx
# Parse JSON into an object with attributes corresponding to dict keys.
bpm = json.load(open(f'{param_path}/backward_params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
bpm.Lx = 2*np.pi*bpm.Lx

# Initialize solver
grid   = ps.Grid1D(fpm)
fsolver = SWHD_1D(fpm)
bsolver = Adjoint_SWHD_1D(bpm)


# true hb
true_hb = np.load(f'{bpm.data_path}/hb.npy')

# Initial conditions
v1 = 0.05
v2 = 0.3927
v3 = 2
uu0 = v1 * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2)
uu0x = - v1 * 2 *( (grid.xx - np.pi/v3)/v2**2 ) * np.exp(-((grid.xx - np.pi/v3) ** 2) / v2 ** 2)

c1 = 0.05
c2 = 0.3927
c3 = 2
hh0 = fpm.h0 + c1 * np.exp(-((grid.xx - np.pi/c3) ** 2) / c2 ** 2)

for iit in range(20):
    uu = uu0
    hh = hh0
    # hb = s1*np.exp(-(grid.xx-np.pi/s3)**2/s2**2) # the true hb
    try:
        hb = np.load(f'{fpm.hb_path}/hb_{iit:00}.npy') # load last ansatz
    except:
        hb = np.zeros_like(grid.xx) # initial ansatz

    fields = [uu, hh, hb]

    # Forward Evolve
    print(f'iit {iit} : evolving forward')
    fsolver.evolve(fields, fpm.T, bstep=fpm.bstep, ostep=fpm.ostep) # last fu, fh, fhb
    print(f'done forward')

    # calculate loss for new fields
    u_loss = np.sum(np.array([(np.load(f'{bpm.data_path}/uu_{tstep:04}.npy') - np.load(f'{bpm.field_path}/uu_{tstep:04}.npy'))**2 for tstep in range(0, int(fpm.T/fpm.dt), 250)]))
    h_loss = np.sum(np.array([(np.load(f'{bpm.data_path}/hh_{tstep:04}.npy') - np.load(f'{bpm.field_path}/hh_{tstep:04}.npy'))**2 for tstep in range(0, int(fpm.T/fpm.dt), 250)]))

    loss = [f'{iit}', f'{u_loss:.6e}' , f'{h_loss:.6e}']
    with open(f'{fpm.hb_path}/loss.dat', 'a') as output:
        print(*loss, file=output)

    hb_val = np.sum((true_hb - hb)**2)
    loss = [f'{iit}', f'{hb_val:.6e}']
    with open(f'{fpm.hb_path}/hb_val.dat', 'a') as output:
        print(*loss, file=output)


    # adjoint equations solver

    # Null initial conditions for adjoint state
    uu_ = np.zeros_like(grid.xx)
    hh_ = np.zeros_like(grid.xx)

    fields = [uu_, hh_]

    # Backward Evolve
    print(f'\niit {iit} : evolving backward')
    fields = bsolver.evolve(fields, bpm.T, bpm.data_path, bpm.field_path, bstep=bpm.bstep, ostep=bpm.ostep) # last fu_, fh_
    print(f'done backward')

    # calculate dg/dhb = h_ * ux at t = 0 (initial time for forward pass)
    print(f'\niit {iit} : calculate dg/hb')
    dg = fields[1] * uu0x
    print('done')

    # update hb values with GD
    print(f'\niit {iit} : update hb')
    hb = hb - bpm.lgd * dg
    print('done')

    # save for the following iteration
    print(f'\niit {iit} : save hb')
    np.save(f'{fpm.hb_path}/hb_{iit+1:00}.npy', hb)
    print('done')

    # Plot fields
    print(f'\niit {iit} : plot')
    tval = int(fpm.T/fpm.dt*0.5)
    out_u = np.load(f'{fpm.out_path}/uu_{tval:04}.npy')
    out_h = np.load(f'{fpm.out_path}/hh_{tval:04}.npy')

    f,axs = plt.subplots(ncols=3, figsize = (15,5))
    axs[0].plot(hb , color = 'blue', label = 'pred hb')
    axs[0].plot(true_hb , alpha = 0.6, color = 'green', label = 'true hb')
    axs[0].legend()
    axs[1].plot(out_u , label = 'u_')
    axs[1].legend()
    axs[2].plot(out_h , label = 'h_')
    axs[2].legend()
    plt.savefig(f'{fpm.hb_path}/fields_{iit:00}.png')

    plt.figure()
    plt.plot(np.sqrt((hb-true_hb)**2) , label = 'hb error')
    plt.savefig(f'{fpm.hb_path}/hb_error_{iit:00}.png')


    loss = np.loadtxt(f'{fpm.hb_path}/loss.dat', unpack=True)
    plt.figure()
    plt.plot(loss[0], loss[1], label = '$(u-\hat{u})^2$')
    plt.plot(loss[0], loss[2], label = '$(h- \hat{h})^2$')
    plt.legend()
    plt.savefig(f'{fpm.hb_path}/loss.png')

    val = np.loadtxt(f'{fpm.hb_path}/hb_val.dat', unpack=True)
    plt.figure()
    plt.plot(val[0], val[1], label = '$(hb-\hat{hb})^2$')
    plt.legend()
    plt.savefig(f'{fpm.hb_path}/hb_val.png')
    print(f'done iit {iit}')
