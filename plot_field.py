import numpy as np
import matplotlib.pyplot as plt
import params2D as pm
import mod2D as mod
import os

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)



def get_var_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable][0] 

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

def saveplot(field, title, path):
    plt.figure(figsize = (13,10))
    plt.imshow(field, origin = 'lower', extent = [0, pm.Nx, 0, pm.Ny])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.savefig(f'{path}/{title}.png')
    plt.close()

def plot_newton_step(nstep):
    fields = np.load(f'output/fields_{nstep}.npz')
    uu = fields['uu']
    vv = fields['vv']
    oz = vort(uu, vv)

    fields = np.load(f'output/fields_{nstep}_T.npz')
    uu2 = fields['uu']
    vv2 = fields['vv']
    oz2 = vort(uu2, vv2)
    print('Newton step = ', nstep)
    print('|u(t)|=', np.linalg.norm(uu))
    print('|u(t+T)|=', np.linalg.norm(uu2))
    print('|u(t+T)-u(t)|=', np.linalg.norm(uu-uu2))

    path = f'plots/nstep{nstep}'

    for folder in ('', '/u', '/v', '/w'):
        mkdir(path+folder)

    saveplot(uu, 'u(t)', path+'/u')
    saveplot(vv, 'v(t)', path+'/v')
    saveplot(oz, 'w(t)', path+'/w')

    saveplot(uu2, 'u(t+T)', path+'/u')
    saveplot(vv2, 'v(t+T)', path+'/v')
    saveplot(oz2, 'w(t+T)', path+'/w')

    saveplot(uu2-uu, 'u(t+T)-u(t)', path+'/u')
    saveplot(vv2-vv, 'v(t+T)-v(t)', path+'/v')
    saveplot(oz2-oz, 'w(t+T)-w(t)', path+'/w')

def get_T(nstep):
    solver = np.loadtxt('prints/solver.txt', delimiter = ',', skiprows = 1)
    return solver[nstep,1]

def save_fields(fu, fv, step, path):
    uu = mod.inverse(fu)
    vv = mod.inverse(fv)
    np.savez(f'{path}/fields_{step:05}.npz', uu=uu,vv=vv)

def evolve(X, T, nstep):
    uu, vv = mod.unflatten_fields(X, pm)
    fu = mod.forward(uu)
    fv = mod.forward(vv)
    # Forcing
    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
    fx = mod.forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = mod.inc_proj(fx, fy, grid)

    for step in range(int(T/pm.dt)):

        # Store previous time step
        fup = np.copy(fu)
        fvp = np.copy(fv)

        if step%pm.fields_freq==0:
            save_fields(fu, fv, step, f'output/nstep{nstep}')
            uu = mod.inverse(fu)
            vv = mod.inverse(fv)
            oz = vort(uu, vv)
            saveplot(uu, f'u(t={step:05})', f'plots/nstep{nstep}/evol')
            saveplot(vv, f'v(t={step:05})', f'plots/nstep{nstep}/evol')
            saveplot(oz, f'w(t={step:05})', f'plots/nstep{nstep}/evol')

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

    uu = mod.inverse(fu)
    vv = mod.inverse(fv)
    X  = mod.flatten_fields(uu, vv)
    return X


def plot_evolution(nstep):
    f_name = f'fields_{nstep}'
    fields = np.load(f'output/{f_name}.npz')
    uu = fields['uu']
    vv = fields['vv']
    X = mod.flatten_fields(uu, vv)
    T = get_T(nstep)
    mkdir(f'output/nstep{nstep}')
    mkdir(f'plots/nstep{nstep}/evol')
    Y = evolve(X, T, nstep)


mkdir('plots')

plot_newton_step(0)
plot_newton_step(8)

plot_evolution(0)
plot_evolution(8)