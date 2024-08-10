import numpy as np

class Grid1D:
    def __init__(self, pm):
        xx, dx = np.linspace(0, pm.L, pm.Nx, endpoint=False, retstep=True)
        tt     = np.arange(0, pm.T, pm.dt)

        ki = np.fft.rfftfreq(pm.Nx, 1/pm.Nx) 
        kx = 2.0*np.pi*np.fft.rfftfreq(pm.Nx, dx) 
        k2 = kx**2
        kk = np.sqrt(k2)
        kr = np.round(kk)

        self.xx = xx
        self.dx = dx
        self.tt = tt
        self.dt = pm.dt

        self.kx = kx
        self.k2 = k2
        self.kk = kk
        self.kr = kr
        self.ki = ki

        self.norm = 1.0/(pm.Nx**2)

        # De-aliasing modes
        self.zero_mode = 0
        self.dealias_modes = (ki > pm.Nx/3)


class Grid2D:
    def __init__(self, pm):
        xi, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
        yi, dy = np.linspace(0, pm.Ly, pm.Ny, endpoint=False, retstep=True)
        xx, yy = np.meshgrid(xi, yi) #should add indexing = 'ij' if Nx and Ny are different for clarity
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

def initial_conditions1D(grid, pm):
    if pm.stat == 0:
        # Initial conditions
        # uu = np.cos(2*np.pi*grid.xx/pm.L)
        uu = (0.3*np.cos(2*np.pi*3.0*grid.xx/pm.L) +
              0.4*np.cos(2*np.pi*5.0*grid.xx/pm.L) +
              0.5*np.cos(2*np.pi*4.0*grid.xx/pm.L) 
              )
        fu = forward(uu)
    else:
        path = pm.path
        f_name = f'fields_{pm.stat:06}'
        fields = np.load(f'{path}output/{f_name}.npz')
        uu = fields['uu']
        fu = forward(uu)
    return [fu]

def initial_conditions2D(grid, pm):
    if pm.stat == 0:
        # Initial conditions
        uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
        vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
        fu = forward(uu)
        fv = forward(vv)
        fu, fv = inc_proj2D(fu, fv, grid)
    else:
        path = '/share/data4/jcullen/pySPEC/run1/'
        f_name = f'fields_{pm.stat:06}'
        fields = np.load(f'{path}output/{f_name}.npz')
        uu = fields['uu']
        vv = fields['vv']
        fu = forward(uu)
        fv = forward(vv)
    return fu, fv

