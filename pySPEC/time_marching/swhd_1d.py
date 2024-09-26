''' 1D Shallow Water Equations '''

import numpy as np

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class SWHD_1D(PseudoSpectral):
    ''' 1D Shallow Water Equations '''

    num_fields = 3
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)

        self.hb = ...

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu  = fields[0]
        fup = prev[0]

        fh  = fields[1]
        fhp = prev[1]

        # Non-linear term
        uu  = self.grid.inverse(fu)
        fu2 = self.grid.forward(uu**2)

        fu = fup + (self.grid.dt/oo) * (
            - (0.5*1.0j*self.grid.kx*fu2)
            - pm.nu*(self.grid.k2)*fu 
            - 1.0j*self.grid.kx*fh
            )

        fh = fhp + (self.grid.dt/oo) * (
            # - (0.5*1.0j*self.grid.kx*fu2)
            )

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0 
        fu[self.grid.dealias_modes] = 0.0 
        fh[self.grid.zero_mode] = 0.0 
        fh[self.grid.dealias_modes] = 0.0 

        return [fu, fh]

    def outs(self, fields, step):
        uu = self.grid.inverse(fields[0])
        np.save(f'uu_{step:04}', uu)

    def balance(self, fields, step):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open('balance.dat', 'a') as output:
            print(*bal, file=output)
