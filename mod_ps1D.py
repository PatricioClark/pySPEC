import numpy as np

class Grid:
    def __init__(self, pm):
        xx, dx = np.linspace(0, pm.L, pm.Nx, endpoint=False, retstep=True)
        tt     = np.arange(0, pm.T, pm.dt)

        kx = np.fft.rfftfreq(pm.Nx, dx) 
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

        self.norm = 1.0/pm.Nx**2

        # De-aliasing modes
        self.zero_mode = 0
        self.dealias_modes = (kk > pm.Nx/3)

def forward(ui):
    ''' Forward Fourier transform '''
    return np.fft.rfft(ui)

def inverse(ui):
    ''' Invserse Fourier transform '''
    return np.fft.irfft(ui).real

def deriv(ui, ki):
    ''' First derivative in ki direction '''
    return 1.0j*ki*ui

def avg(ui, grid):
    ''' Mean in Fourier space '''
    return grid.norm * np.sum(ui)

def initial_conditions(grid, pm):
    if pm.stat == 0:
        # Initial conditions
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

