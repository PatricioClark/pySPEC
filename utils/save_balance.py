
import numpy as np
import matplotlib.pyplot as plt
import params2D as pm
import mod2D as mod

grid = mod.Grid(pm)

path = '/share/data4/jcullen/pySPEC/run1/'
fields = np.load(f'{path}output2/fields_t.npz')
uu_t = fields['uu_t']
vv_t = fields['vv_t']


for i in range(len(uu_t[:,0,0])):
    output_n = 2
    start_step = 500_000 #extraído de save_fields_t local
    save_freq = 50
    step = int(start_step + round(i*save_freq))
    dir_name = 'balance'

    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
    fx = mod.forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = mod.inc_proj(fx, fy, grid)

    uu = uu_t[i,:,:]
    vv = vv_t[i,:,:]
    fu = mod.forward(uu)
    fv = mod.forward(vv)
    u2 = mod.inner(fu, fv, fu, fv)
    eng = 0.5 * mod.avg(u2, grid)
    inj = mod.avg(mod.inner(fu, fv, fx, fy), grid)
    ens = - pm.nu * mod.avg(grid.k2*u2, grid)

    with open(f'{dir_name}/balance{output_n}.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {ens}, {inj}', file=ff) 

path = '/share/data4/jcullen/pySPEC/run1/'
fields = np.load(f'{path}output3/fields_t.npz')
uu_t = fields['uu_t']
vv_t = fields['vv_t']


for i in range(len(uu_t[:,0,0])):
    output_n = 3
    start_step = 2_000_000 #extraído de save_fields_t local
    save_freq = 100
    step = int(start_step + round(i*save_freq))
    dir_name = 'balance'

    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
    fx = mod.forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = mod.inc_proj(fx, fy, grid)

    uu = uu_t[i,:,:]
    vv = vv_t[i,:,:]
    fu = mod.forward(uu)
    fv = mod.forward(vv)
    u2 = mod.inner(fu, fv, fu, fv)
    eng = 0.5 * mod.avg(u2, grid)
    inj = mod.avg(mod.inner(fu, fv, fx, fy), grid)
    ens = - pm.nu * mod.avg(grid.k2*u2, grid)

    with open(f'{dir_name}/balance{output_n}.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {ens}, {inj}', file=ff) 
