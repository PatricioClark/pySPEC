'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import numpy as np
import pySPEC.mods.mod_ps1D as mod

from .solver import Solver

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
                uu  = mod.inverse(fu)
                fu2 = mod.forward(uu**2)

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
        eng = mod.energy(fields, self.grid)
        return [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
