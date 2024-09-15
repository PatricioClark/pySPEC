import numpy as np

class Grid1D:
    def __init__(self, pm):
        xx, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
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

        # Norm and de-aliasing
        self.norm = 1.0/(pm.Nx**2)
        self.zero_mode = 0
        self.dealias_modes = (ki > pm.Nx/3)

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

        ky = 2.0*np.pi*np.fft.rfftfreq(pm.Ny, dy)
        k2 = self.kx**2 + ky**2
        kk = np.sqrt(k2)
        kr = np.round(kk)

        self.xx = xx
        self.yy = yy
        self.dy = dy

        self.k2 = k2
        self.ky = ky
        self.kk = kk
        self.kr = kr

        # Norm and de-aliasing
        self.norm = 1.0/(pm.Nx**2*pm.Ny**2)
        self.zero_mode = (0, 0)
        self.dealias_modes = (kk > pm.Nx/3)

        # Solenoidal mode proyector
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1.0 - self.kx**2/k2)
            self.pyy = np.nan_to_num(1.0 - self.ky**2/k2)
            self.pxy = np.nan_to_num(- self.kx*self.ky/k2)

    def inc_proj2D(self, fields):
        ''' Project onto solenoidal modes '''
        fu = fields[0]
        fv = fields[1]
        return self.pxx*fu + self.pxy*fv, self.pxy*fu + self.pyy*fv
