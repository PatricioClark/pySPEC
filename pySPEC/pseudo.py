import numpy as np

class Grid:
    def __init__(self, pm):
        xx, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
        tt     = np.arange(0, pm.T, pm.dt)

        if pm.dim == 2:
            xi, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
            yi, dy = np.linspace(0, pm.Ly, pm.Ny, endpoint=False, retstep=True)
            xx, yy = np.meshgrid(xi, yi, indexing='ij')
        else:
            dy = 0.0
            yy = None

        ki = np.fft.rfftfreq(pm.Nx, 1/pm.Nx) 
        kx = 2.0*np.pi*np.fft.rfftfreq(pm.Nx, dx) 
        k2 = kx**2
        if pm.dim == 2:
            ky = 2.0*np.pi*np.fft.rfftfreq(pm.Ny, dy)
            k2 = k2 + ky**2
        else:
            ky = 0.0
        kk = np.sqrt(k2)
        kr = np.round(kk)

        self.xx = xx
        self.dx = dx
        self.tt = tt
        self.dt = pm.dt
        if pm.dim == 2:
            self.yy = yy
            self.dy = dy

        self.kx = kx
        self.k2 = k2
        self.kk = kk
        self.kr = kr
        self.ki = ki

        # Norm and de-aliasing
        self.norm = 1.0/(pm.Nx**2)
        self.zero_mode = 0
        self.dealias_modes = (ki > pm.Nx/3)

        if pm.dim == 2:
            self.zero_mode = (0, 0)
            self.dealias_modes = (kk > pm.Nx/3)

            # Solenoidal mode proyector
            with np.errstate(divide='ignore', invalid='ignore'):
                self.pxx = np.nan_to_num(1.0 - kx**2/k2)
                self.pyy = np.nan_to_num(1.0 - ky**2/k2)
                self.pxy = np.nan_to_num(- kx*ky/k2)

def forward(ui):
    ''' Forward Fourier transform '''
    return np.fft.rfftn(ui)

def inverse(ui):
    ''' Invserse Fourier transform '''
    return np.fft.irfftn(ui).real

def deriv(ui, ki):
    ''' First derivative in ki direction '''
    return 1.0j*ki*ui

def avg(ui, grid):
    ''' Mean in Fourier space '''
    return 2.0 * grid.norm * np.sum(ui)

def inner(a, b):
    ''' Inner product '''
    prod = 0.0
    for ca, cb in zip(a, b):
        prod += (ca*cb.conjugate()).real
    return prod

def energy(fields, grid):
    u2  = inner(fields, fields)
    eng = 0.5*avg(u2, grid)
    return eng

def inc_proj2D(fu, fv, grid):
    ''' Project onto solenoidal modes '''
    return grid.pxx*fu + grid.pxy*fv, grid.pxy*fu + grid.pyy*fv
