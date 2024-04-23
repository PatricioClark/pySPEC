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

        # De-aliasing modes
        self.zero_mode = (0, 0)
        self.dealias_modes = (kk > pm.Nx/3)

        # Solenoidal mode proyector
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1 - kx**2/k2)
            self.pyy = np.nan_to_num(1 - ky**2/k2)
            self.pxy = np.nan_to_num(- kx*ky/k2)

def forward(ui):
    return np.fft.fft2(ui)

def inverse(ui):
    return np.fft.ifft2(ui).real

def deriv(ui, ki):
    return 1.0j*ki*ui

def balance(fu, fv, fx, fy, grid, pm, step):
    norm = 1.0/(pm.Nx*pm.Ny)**2
    u2 = fu*fu.conjugate() + fv*fv.conjugate()
    eng = 0.5 * norm * np.sum(u2).real
    inj = norm * np.sum(fx*fu.conjugate() + fy*fv.conjugate()).real
    ens = - pm.nu * norm * np.sum(grid.k2*u2).real

    with open('balance.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {inj}, {ens}', file=ff) 

def check(fu, fv, fx, fy, grid, pm, step):
    norm = 1.0/(pm.Nx*pm.Ny)**2
    u2 = fu*fu.conjugate() + fv*fv.conjugate()

    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    oz = uy - vx

    plt.figure(3)
    plt.imshow(oz)
    plt.colorbar()
    plt.savefig(f'vor_{step:06}.png')

    plt.figure(5)
    uk = np.array([np.sum(u2[np.where(grid.kr == ki)]).real for ki in range(pm.Nx//2)])
    plt.loglog(0.5 * norm * uk)
    plt.savefig(f'uk_{step:06}.png')

    plt.close('all')
