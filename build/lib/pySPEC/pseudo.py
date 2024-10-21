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
        ''' Inverse Fourier transform '''
        return np.fft.irfftn(ui).real

    @staticmethod
    def deriv(ui, ki):
        ''' First derivative in ki direction '''
        return 1.0j*ki*ui

    def avg(self, ui):
        ''' Mean in Fourier space '''
        return 2.0 * self.norm * np.sum(ui)

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

        # Norm, de-aliasing and solenoidal mode proyector
        self.norm = 1.0/(pm.Nx**2*pm.Ny**2)
        self.zero_mode = (0, 0)
        self.dealias_modes = (kr > 1/9)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1.0 - self.kx**2/k2)
            self.pyy = np.nan_to_num(1.0 - self.ky**2/k2)
            self.pxy = np.nan_to_num(- self.kx*self.ky/k2)

    def inc_proj(self, fields):
        ''' Project onto solenoidal modes '''
        fu = fields[0]
        fv = fields[1]
        return self.pxx*fu + self.pxy*fv, self.pxy*fu + self.pyy*fv
