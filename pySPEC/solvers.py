'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import abc
import numpy as np

from . import pseudo as ps

class Solver(abc.ABC):
    def __init__(self, pm):
        self.grid = ps.Grid1D(pm)
        self.pm = pm

    def evolve(self, fields, T, bstep=None, sstep=None, ostep=None):
        ''' Evolves velocity fields to time T '''

        Nt = round(T/self.pm.dt)
        for step in range(Nt):
            # Store previous time step
            prev = np.copy(fields)

            # Time integration
            for oo in range(self.pm.rkord, 0, -1):
                fields = self.rkstep(fields, prev, oo)

            # Write outputs
            self.write_outputs(fields, step, bstep, sstep, ostep)

        return fields

    @abc.abstractmethod
    def rkstep(self, fields, prev, oo):
        return []

    def balance(self, fields, step):
        eng = ps.energy(fields, self.grid)
        return [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']

    def spectra(self, fields, step):
        pass

    def outs(self, fields, step):
        pass

    def write_outputs(self, fields, step, bstep, sstep, ostep):
        if bstep is not None and step%bstep==0:
            bal = self.balance(fields, step)
            with open('balance.dat', 'a') as output:
                print(*bal, file=output)

        if sstep is not None and step%sstep==0:
            self.spectra(fields, step)
            
        if ostep is not None and step%ostep==0:
            self.outs(fields, step)

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

class KuramotoSivashinsky(Solver):
    ''' 1D Kuramoto Sivashinsky equation '''
    def __init__(self, pm):
        self.grid = ps.Grid1D(pm)
        self.pm   = pm

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu  = fields[0]
        fup = prev[0]
        # Non-linear term
        uu  = ps.inverse(fu)
        fu2 = ps.forward(uu**2)

        fu = fup + (self.grid.dt/oo) * (
            - (0.5*1.0j*self.grid.kx*fu2)
            + ((self.grid.k2)*fu) 
            - ((self.grid.k2**2)*fu)
            )

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0 
        fu[self.grid.dealias_modes] = 0.0 

        return [fu]

    def outs(self, fields, step):
        uu = ps.inverse(fields[0])
        np.save(f'uu_{step:04}', uu)
