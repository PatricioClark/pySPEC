import numpy as np

class Grid:
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

def inner(a, b):
    ''' Inner product '''
    prod = 0.0
    for ca, cb in zip(a, b):
        prod += (ca*cb.conjugate()).real
    return prod

def inc_proj(fu, fv, grid):
    ''' Project onto solenoidal modes '''
    return grid.pxx*fu + grid.pxy*fv, grid.pxy*fu + grid.pyy*fv

def vort(uu, vv, pm, grid):
    ''' Computes vorticity field '''
    fu, fv = forward(uu), forward(vv)
    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    if not pm.fvort:
        return uy - vx
    else:
        return forward(uy-vx)

def inv_vort(oz, pm, grid):
    ''' Computes uu,vv from oz vorticity'''
    if not pm.fvort:
        foz = forward(oz)
    else:
        foz = oz
    fu = np.divide(-deriv(foz, grid.ky), grid.k2, out = np.zeros_like(foz), where = grid.k2!=0.)
    fv = np.divide(deriv(foz, grid.kx), grid.k2, out = np.zeros_like(foz), where = grid.k2!=0.)
    uu, vv = inverse(fu), inverse(fv)
    return uu, vv

def save_oz_ek(oz_t, Ek_t, fu, fv, grid, pm, idx):
    ''' Save vorticity and spectrum '''
    #vorticity
    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    oz = uy - vx

    #spectrum
    u2 = inner(fu, fv, fu, fv)
    uk = np.array([np.sum(u2[np.where(grid.kr == ki)]) for ki in range(pm.Nx//2)])
    Ek = 0.5 * grid.norm * uk

    oz_t[idx, :,:] = oz
    Ek_t[idx, :] = Ek

def balance(fu, fv, fx, fy, grid, pm, i_newt, step):
    ''' Save global quantities: energy, dissipation, injection rate '''
    u2 = inner(fu, fv, fu, fv)
    eng = 0.5 * avg(u2, grid)
    dis = - pm.nu * avg(grid.k2*u2, grid)
    inj = avg(inner(fu, fv, fx, fy), grid)

    with open(f'balance/balance{i_newt}.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {dis}, {inj}', file=ff) 

def check(fu, fv, grid, pm, step, nstep):
    u2 = inner(fu, fv, fu, fv)

    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    oz = uy - vx

    uk = np.array([np.sum(u2[np.where(grid.kr == ki)]) for ki in range(pm.Nx//2)])
    Ek = 0.5 * grid.norm * uk

    np.save(f'Ek/nstep{nstep}/Ek.{step:06}', Ek)    
    np.save(f'evol/nstep{nstep}/oz.{step:06}', oz)    


def initial_conditions(grid, pm):
    if pm.stat == 0:
        # Initial conditions
        uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
        vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
        fu = forward(uu)
        fv = forward(vv)
        fu, fv = inc_proj(fu, fv, grid)
    else:
        path = '/share/data4/jcullen/pySPEC/run1/'
        f_name = f'fields_{pm.stat:06}'
        fields = np.load(f'{path}output/{f_name}.npz')
        uu = fields['uu']
        vv = fields['vv']
        fu = forward(uu)
        fv = forward(vv)
    return fu, fv

