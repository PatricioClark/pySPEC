'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import numpy as np
import mod_ps2D as mod

import abc

class Solver(abc.ABC):
    @abc.abstractmethod
    def evolve(self, fields, T):
        pass

    @abc.abstractmethod
    def check_balance(self, fields, tt):
        pass

    @abc.abstractmethod
    def write_spectra(self, fields, tt):
        pass

    @abc.abstractmethod
    def write_fields(self, fields, tt):
        pass

class KolmogorovFlow(Solver):
    '''
    Kolmogorov flow: 2D Navier-Stokes with fx = sin(2*pi*kf*y/Ly) forcing.

    See Eq. (6.143) in Pope's Turbulent flows for details on the Fourier
    decomposition of the NS equations and the pressure proyector.
    '''
    def __init__(self, grid, pm, kf=4):
        self.grid = grid
        self.pm = pm

        # Forcing
        self.kf = kf
        self.fx = np.sin(2*np.pi*kf*grid.yy/pm.Ly)
        self.fx = mod.forward(self.fx)
        self.fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
        self.fx, self.fy = mod.inc_proj(self.fx, self.fy, grid)

    def evolve(self, fields, T):
        ''' Evolves velocity fields to time T '''
        fu, fv = fields

        Nt = round(T/self.pm.dt)
        for step in range(Nt):
            # Store previous time step
            fup = np.copy(fu)
            fvp = np.copy(fv)
       
            # Time integration
            for oo in range(self.pm.rkord, 0, -1):
                # Non-linear term
                uu = mod.inverse(fu)
                vv = mod.inverse(fv)
                
                ux = mod.inverse(mod.deriv(fu, self.grid.kx))
                uy = mod.inverse(mod.deriv(fu, self.grid.ky))

                vx = mod.inverse(mod.deriv(fv, self.grid.kx))
                vy = mod.inverse(mod.deriv(fv, self.grid.ky))

                gx = mod.forward(uu*ux + vv*uy)
                gy = mod.forward(uu*vx + vv*vy)
                gx, gy = mod.inc_proj(gx, gy, self.grid)

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
