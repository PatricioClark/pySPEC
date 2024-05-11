import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, pm):
        xi, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
        yi, dy = np.linspace(0, pm.Ly, pm.Ny, endpoint=False, retstep=True)
        xx, yy = np.meshgrid(xi, yi)
        tt, dt = np.linspace(0, pm.T, pm.Nt, endpoint=False, retstep=True)

        kx = np.fft.fftfreq(pm.Nx, dx) 
        ky = np.fft.fftfreq(pm.Ny, dy) 
        kx, ky = np.meshgrid(kx, ky)
        k2 = kx**2 + ky**2
        kk = np.sqrt(k2)
        kr = np.round(kk)

        self.xx = xx
        self.dx = dx
        self.yy = yy
        self.dy = dy
        self.tt = tt
        self.dt = dt

        self.kx = kx
        self.ky = ky
        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.norm = 1.0/(pm.Nx*pm.Ny)**2

        # De-aliasing modes
        self.zero_mode = (0, 0)
        self.dealias_modes = (kk > pm.Nx/3)

        # Solenoidal mode proyector
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1.0 - kx**2/k2)
            self.pyy = np.nan_to_num(1.0 - ky**2/k2)
            self.pxy = np.nan_to_num(- kx*ky/k2)

def forward(ui):
    ''' Forward Fourier transform '''
    return np.fft.fft2(ui)

def inverse(ui):
    ''' Invserse Fourier transform '''
    return np.fft.ifft2(ui).real

def deriv(ui, ki):
    ''' First derivative in ki direction '''
    return 1.0j*ki*ui

def avg(ui, grid):
    ''' Mean in Fourier space '''
    return grid.norm * np.sum(ui)

def inner(a, b, c, d):
    ''' Inner product '''
    return (a*c.conjugate() + b*d.conjugate()).real

def inc_proj(fu, fv, grid):
    ''' Project onto solenoidal modes '''
    return grid.pxx*fu + grid.pxy*fv, grid.pxy*fu + grid.pyy*fv

def balance(fu, fv, fx, fy, grid, pm, step):
    u2 = inner(fu, fv, fu, fv)
    eng = 0.5 * avg(u2, grid)
    inj = avg(inner(fu, fv, fx, fy), grid)
    ens = - pm.nu * avg(grid.k2*u2, grid)

    with open('balance.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {inj}, {ens}', file=ff) 

def check(fu, fv, fx, fy, grid, pm, step):
    u2 = inner(fu, fv, fu, fv)

    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    oz = uy - vx

    plt.figure(3)
    plt.imshow(oz)
    plt.colorbar()
    plt.savefig(f'plots/vor_{step:06}.png')

    plt.figure(5)
    uk = np.array([np.sum(u2[np.where(grid.kr == ki)]) for ki in range(pm.Nx//2)])
    plt.loglog(0.5 * grid.norm * uk)
    plt.savefig(f'plots/uk_{step:06}.png')

    plt.close('all')

def save_fields(fu, fv, step):
    uu = inverse(fu)
    vv = inverse(fv)
    np.savez(f'output/fields_{step:06}.npz', uu=uu,vv=vv,fu=fu,fv=fv)

def flatten_fields(uu, vv):
    return np.concatenate((uu.flatten(), vv.flatten()))

def unflatten_fields(X, pm):
    ll = len(X)//2
    uu = X[:ll].reshape((pm.Nx, pm.Ny))
    vv = X[ll:].reshape((pm.Nx, pm.Ny))
    return uu, vv

def evolution_function(grid, pm):
    # Forcing
    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
    fx = forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = inc_proj(fx, fy, grid)

    def evolve(X, T):
        uu, vv = unflatten_fields(X, pm)
        fu = forward(uu)
        fv = forward(vv)
        for step in range(1, int(T/pm.dt)):

            # Store previous time step
            fup = np.copy(fu)
            fvp = np.copy(fv)

            # Time integration
            for oo in range(pm.rkord, 0, -1):
                # Non-linear term
                uu = inverse(fu)
                vv = inverse(fv)
                
                ux = inverse(deriv(fu, grid.kx))
                uy = inverse(deriv(fu, grid.ky))

                vx = inverse(deriv(fv, grid.kx))
                vy = inverse(deriv(fv, grid.ky))

                gx = forward(uu*ux + vv*uy)
                gy = forward(uu*vx + vv*vy)
                gx, gy = inc_proj(gx, gy, grid)

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

        uu = inverse(fu)
        vv = inverse(fv)
        X  = flatten_fields(uu, vv)
        return X
    return evolve

def application_function(evolve, grid, pm):
    def apply_A(X, Y, T, dX, dT):
        epsilon = 1e-7
        deriv_x0 = evolve(X+dX, T) - Y
        deriv_x0 = deriv_x0/epsilon

        rhs = evolve(X, pm.dt) - X
        rhs = rhs/pm.dt

        return np.concatenate([deriv_x0-dX, rhs*dT])
    return apply_A
