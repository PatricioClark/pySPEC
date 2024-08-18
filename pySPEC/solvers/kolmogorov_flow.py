''' 2D Kolmogorov flow solver '''

import numpy as np

from .solver import Solver
from .. import pseudo as ps


class KolmogorovFlow(Solver):
    '''
    Kolmogorov flow: 2D Navier-Stokes with fx = sin(2*pi*kf*y/Ly) forcing.

    nu = 1/Re

    See Eq. (6.143) in Pope's Turbulent flows for details on the Fourier
    decomposition of the NS equations and the pressure proyector.
    '''
    def __init__(self, pm, kf=4):
        self.grid = ps.Grid2D(pm)
        self.pm = pm

        # Forcing
        self.kf = kf
        self.fx = np.sin(2*np.pi*kf*self.grid.yy/pm.Ly)
        self.fx = ps.forward(self.fx)
        self.fy = np.zeros_like(self.fx, dtype=complex)
        self.fx, self.fy = ps.inc_proj2D(self.fx, self.fy, self.grid)

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu, fv = fields
        fup, fvp = prev

        # Non-linear term
        uu = ps.inverse(fu)
        vv = ps.inverse(fv)
        
        ux = ps.inverse(ps.deriv(fu, self.grid.kx))
        uy = ps.inverse(ps.deriv(fu, self.grid.ky))

        vx = ps.inverse(ps.deriv(fv, self.grid.kx))
        vy = ps.inverse(ps.deriv(fv, self.grid.ky))

        gx = ps.forward(uu*ux + vv*uy)
        gy = ps.forward(uu*vx + vv*vy)
        gx, gy = ps.inc_proj2D(gx, gy, self.grid)

        # Equations
        fu = fup + (self.grid.dt/oo) * (
            - gx
            - self.pm.nu * self.grid.k2 * fu 
            + self.fx
            )

        fv = fvp + (self.grid.dt/oo) * (
            - gy
            - self.pm.nu * self.grid.k2 * fv 
            + self.fy
            )

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0 
        fv[self.grid.zero_mode] = 0.0 
        fu[self.grid.dealias_modes] = 0.0 
        fv[self.grid.dealias_modes] = 0.0

        return [fu, fv]

