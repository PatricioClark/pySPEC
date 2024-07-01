import numpy as np
import matplotlib.pyplot as plt
import params2D as pm
import mod2D as mod
import os
import subprocess

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)

def vort(uu, vv):
    fu, fv = mod.forward(uu), mod.forward(vv)
    uy = mod.inverse(mod.deriv(fu, grid.ky))
    vx = mod.inverse(mod.deriv(fv, grid.kx))
    return uy - vx

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

def mkplotdirs(nstep, dir_name):
    for folder in ('evol',  'Ek'):
        mkdir(f'../{dir_name}/{folder}')
        mkdir(f'../{dir_name}/{folder}/nstep{nstep}')

def get_T(nstep, dir_name):
    solver = np.loadtxt(f'../{dir_name}/prints/solver.txt', delimiter = ',', skiprows = 1)
    return solver[nstep,1]

def save_fields(fu, fv, step, path):
    uu = mod.inverse(fu)
    vv = mod.inverse(fv)
    np.savez(f'{path}/fields_{step:05}.npz', uu=uu,vv=vv)

def check(fu, fv, grid, pm, step, nstep, dir_name):
    u2 = mod.inner(fu, fv, fu, fv)

    uy = mod.inverse(mod.deriv(fu, grid.ky))
    vx = mod.inverse(mod.deriv(fv, grid.kx))
    oz = uy - vx

    uk = np.array([np.sum(u2[np.where(grid.kr == ki)]) for ki in range(pm.Nx//2)])
    Ek = 0.5 * grid.norm * uk

    np.save(f'../{dir_name}/Ek/nstep{nstep}/Ek.{step:06}', Ek)    
    np.save(f'../{dir_name}/evol/nstep{nstep}/oz.{step:06}', oz)    


def balance(fu, fv, fx, fy, grid, pm, step, nstep, dir_name):
    u2 = mod.inner(fu, fv, fu, fv)
    eng = 0.5 * mod.avg(u2, grid)
    inj = mod.avg(mod.inner(fu, fv, fx, fy), grid)
    ens = - pm.nu * mod.avg(grid.k2*u2, grid)

    with open(f'../{dir_name}/balance{nstep}.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {ens}, {inj}', file=ff) 




def evolve(nstep, dir_name):
    '''Evolve field from nstep newton iter. and save balance, some fields, and Ek '''
    f_name = f'fields_{nstep}' 
    fields = np.load(f'../{dir_name}/output/{f_name}.npz') #Get initial field
    uu = fields['uu']
    vv = fields['vv']
    X = mod.flatten_fields(uu, vv, pm, grid)
    T = get_T(nstep, dir_name)#Get Period corresponding to that newton iter.
    Nt = int(T/pm.dt)

    len_fields = 40
    fields_freq = Nt//len_fields #a total amount of len_fields are saved
    print_freq = 50

    #Evolve initial field
    uu, vv = mod.unflatten_fields(X, pm, grid)
    fu = mod.forward(uu)
    fv = mod.forward(vv)
    # Forcing
    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
    fx = mod.forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = mod.inc_proj(fx, fy, grid)

    with open(f'../{dir_name}/balance{nstep}.dat','w') as ff:
        print(f't, eng, ens, inj', file=ff) 

    for step in range(Nt+1):#extra step to save fields in final state (save prior to evol)

        # Store previous time step
        fup = np.copy(fu)
        fvp = np.copy(fv)

        #save fields every fields_freq and save last field
        if (step%fields_freq==0) or (step == range(Nt+1)[-1]):
            check(fu, fv, grid, pm, step, nstep, dir_name)
        if (step%print_freq==0) or (step == range(Nt+1)[-1]):
            balance(fu, fv, fx, fy, grid, pm, step, nstep, dir_name)

        # Time integration
        for oo in range(pm.rkord, 0, -1):
            # Non-linear term
            uu = mod.inverse(fu)
            vv = mod.inverse(fv)
        
            ux = mod.inverse(mod.deriv(fu, grid.kx))
            uy = mod.inverse(mod.deriv(fu, grid.ky))

            vx = mod.inverse(mod.deriv(fv, grid.kx))
            vy = mod.inverse(mod.deriv(fv, grid.ky))

            gx = mod.forward(uu*ux + vv*uy)
            gy = mod.forward(uu*vx + vv*vy)
            gx, gy = mod.inc_proj(gx, gy, grid)

            # Equations
            fu = fup + (grid.dt/oo) * (
                - gx
                - pm.nu * grid.k2 * fu 
                + fx
                )

            fv = fvp + (grid.dt/oo) * (
                - gy
                - pm.nu * grid.k2 * fv 
                + fy
                )

            # de-aliasing
            fu[grid.zero_mode] = 0.0 
            fv[grid.zero_mode] = 0.0 
            fu[grid.dealias_modes] = 0.0 
            fv[grid.dealias_modes] = 0.0


#Get info from txt with convergence info
orbs, bs_min, iters_min, _ = np.loadtxt('../data_orbs.txt', delimiter = ',', skiprows=1, unpack = True)

for orb_n, b_min, iter_min in zip(orbs, bs_min, iters_min):
    # #Classify by convergence of orbits
    # if b_min < 0.085:

    #Get info from txt with initial orbits info
    data = np.loadtxt('../ginput_data_orbs.txt', delimiter=',', skiprows=1, comments='-', usecols = (0,1,2))
    orb = data[int(orb_n)] #selecciona fila con info de orb_n
    dir_name = f'orb{int(orb[0]):02}_{orb[1]}' #directory with name orb09_5.9 where 09 is the orb nmb and 5.9 the guessed T

    # #make plots for initial state and converged state
    # for nstep in (0, int(iter_min)):
    #     mkplotdirs(nstep,dir_name)
    #     evolve(nstep,dir_name)
