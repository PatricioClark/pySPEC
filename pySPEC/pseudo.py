''' Definitions of 1- and 2-D periodic grids '''
import numpy as np

class Grid1D:
    def __init__(self, pm):
        xx, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
        tt     = np.arange(0, pm.T, pm.dt)

        ki = np.fft.rfftfreq(pm.Nx, 1/pm.Nx).astype(int)
        kx = 2.0*np.pi*np.fft.rfftfreq(pm.Nx, dx) 
        k2 = kx**2
        kk = np.sqrt(k2)

        self.xx = xx
        self.dx = dx
        self.tt = tt
        self.dt = pm.dt

        self.kx = kx
        self.k2 = k2
        self.kk = kk
        self.kr = ki

        self.N = pm.Nx
        self.shape = (pm.Nx,)

        # Norm and de-aliasing
        self.norm = 1.0/(pm.Nx**2)
        self.zero_mode = 0
        self.dealias_modes = (self.kr > pm.Nx/3)

    @staticmethod
    def forward(ui):
        ''' Forward Fourier transform '''
        return np.fft.rfftn(ui)

    @staticmethod
    def inverse(ui):
        ''' Invserse Fourier transform '''
        return np.fft.irfftn(ui).real

    @staticmethod
    def deriv(ui, ki):
        ''' First derivative in ki direction '''
        return 1.0j*ki*ui

    def avg(self, ui):
        ''' Mean in Fourier space. rfft is used, so middle modes must be doubled to account '''
        ''' for negative frequencies. If n is even the last mode contains +fs/2 and -fs/2'''
        tmp = 1 if len(ui) % 2 == 0 else 2
        sum_ui = ui[0] + 2.0*np.sum(ui[1:-1]) + tmp* ui[-1]
        return self.norm * sum_ui

    @staticmethod
    def inner(a, b):
        ''' Inner product '''
        prod = 0.0
        for ca, cb in zip(a, b):
            prod += (ca*cb.conjugate()).real
        return prod

    def energy(self, fields):
        u2  = self.inner(fields, fields)
        eng = 0.5*self.avg(u2)
        return eng

    def enstrophy(self, fields):
        u2  = self.inner(fields, fields)
        ens = 0.5*self.avg(self.k2*u2)
        return ens

    def translate(self, fields, sx):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff*np.exp(1.0j*self.kx*sx) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def deriv_fields(self, fields):
        ''' Compute derivatives in Fourier space '''
        f = [self.forward(ff) for ff in fields]
        f = [self.deriv(ff, self.kx) for ff in f]
        return [self.inverse(ff) for ff in f]


class Grid2D(Grid1D):
    def __init__(self, pm):
        super().__init__(pm)

        xi, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
        yi, dy = np.linspace(0, pm.Ly, pm.Ny, endpoint=False, retstep=True)
        xx, yy = np.meshgrid(xi, yi, indexing='ij')

        kx = 2.0*np.pi*np.fft.fftfreq(pm.Nx, dx) 
        ky = 2.0*np.pi*np.fft.rfftfreq(pm.Ny, dy)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        kk = np.sqrt(k2)

        ki = np.fft.fftfreq(pm.Nx, 1/pm.Nx)
        kj = np.fft.rfftfreq(pm.Ny, 1/pm.Ny)
        ki, kj = np.meshgrid(ki, kj, indexing='ij')
        kr = (ki/pm.Nx)**2 + (kj/pm.Ny)**2

        self.xx = xx
        self.yy = yy
        self.dy = dy

        self.kx = kx
        self.ky = ky
        self.ki = ki
        self.kj = kj
        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.N = pm.Nx*pm.Ny
        self.shape = (pm.Nx, pm.Ny)


        # Norm, de-aliasing and solenoidal mode proyector
        self.norm = 1.0/(pm.Nx**2*pm.Ny**2)
        self.zero_mode = (0, 0)
        self.dealias_modes = (kr > 1/9)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1.0 - self.kx**2/k2)
            self.pyy = np.nan_to_num(1.0 - self.ky**2/k2)
            self.pxy = np.nan_to_num(- self.kx*self.ky/k2)

    def translate2D(self, fields, sx, sy):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff *np.exp(1.0j*self.kx*sx) *np.exp(1.0j*self.ky*sy) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def inc_proj(self, fields):
        ''' Project onto solenoidal modes '''
        fu = fields[0]
        fv = fields[1]
        return self.pxx*fu + self.pxy*fv, self.pxy*fu + self.pyy*fv

    def avg(self, ui):
        ''' Mean in Fourier space. rfft is used, so middle modes must be doubled to account
            for negative frequencies. If n is even the last mode contains +fs/2 and -fs/2'''
        tmp = 1 if len(ui) % 2 == 0 else 2
        sum_ui = np.sum(ui[:,0]) + 2.0*np.sum(ui[:,1:-1]) + tmp* np.sum(ui[:,-1])
        return self.norm * sum_ui

class Grid2D_semi(Grid1D):
    ''' 2D grid periodic only in the horizontal direction. To be used with the SPECTER wrapper '''
    def __init__(self, pm):
        super().__init__(pm)
        zi, dz = np.linspace(0, pm.Lz, pm.Nz, endpoint=False, retstep=True)
        kx, zz = np.meshgrid(self.kx, zi, indexing='ij')

        self.zi = zi
        self.dz = dz
        self.zz = zz
        self.kx = kx

        self.N = pm.Nx*pm.Nz
        self.shape = (pm.Nx, pm.Nz)

    @staticmethod
    def forward(ui):
        ''' Forward Fourier transform '''
        return np.fft.rfft(ui, axis = 0)

    @staticmethod
    def inverse(ui):
        ''' Invserse Fourier transform '''
        ui =  np.fft.irfft(ui, axis = 0).real
        return ui
