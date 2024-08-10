'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import abc
import numpy as np

import pseudo as ps

class Solver(abc.ABC):
    @abc.abstractmethod
    def evolve(self, fields, T, bstep=None, sstep=None, fstep=None):
        return []

    def balance(self, fields, step):
        return []

    def spectra(self, fields, step):
        pass

    def fields(self, fields, step):
        pass

    def write_outputs(self, fields, step, bstep, sstep, fstep):
        if bstep is not None and step%bstep==0:
            bal = self.balance(fields, step)
            with open('balance.dat', 'a') as output:
                print(*bal, file=output)

        if sstep is not None and step%sstep==0:
            self.spectra(fields, step)
            
        if fstep is not None and step%fstep==0:
            self.fields(fields, step)

class KolmogorovFlow(Solver):
    '''
    Kolmogorov flow: 2D Navier-Stokes with fx = sin(2*pi*kf*y/Ly) forcing.

    nu = 1/Re

    See Eq. (6.143) in Pope's Turbulent flows for details on the Fourier
    decomposition of the NS equations and the pressure proyector.
    '''
    def __init__(self, grid, pm, kf=4):
        self.grid = grid
        self.pm = pm

        # Forcing
        self.kf = kf
        self.fx = np.sin(2*np.pi*kf*grid.yy/pm.Ly)
        self.fx = ps.forward(self.fx)
        self.fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
        self.fx, self.fy = ps.inc_proj2D(self.fx, self.fy, grid)

    def evolve(self, fields, T, bstep=None, sstep=None, fstep=None):
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

            # Write outputs
            self.write_outputs(fields, step, bstep, sstep, fstep)

        return [fu, fv]

class KuramotoSivashinsky(Solver):
    ''' 1D Kuramoto Sivashinsky equation '''
    def __init__(self, grid, pm):
        self.grid = grid
        self.pm   = pm

    def evolve(self, fields, T, bstep=None, sstep=None, fstep=None):
        ''' Evolves velocity fields to time T '''
        fu = fields[0]

        Nt = round(T/self.pm.dt)
        for step in range(Nt):
            # Store previous time step
            fup = np.copy(fu)

            # Time integration
            for oo in range(self.pm.rkord, 0, -1):
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

            # Write outputs
            self.write_outputs(fields, step, bstep, sstep, fstep)

        return [fu]

    def balance(self, fields, step):
        eng = ps.energy(fields, self.grid)
        return [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
