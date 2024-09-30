''' 1D Shallow Water Equations '''

import numpy as np

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class SWHD_1D(PseudoSpectral):
    ''' 1D Shallow Water Equations 
        ut + u ux + g hx = 0
        ht + ( u(h-hb) )x = 0
    where u,h are velocity and height fields,
    and hb is bottom topography condition.
    '''

    num_fields = 2
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

        hb = self.hb # real function

        # Non-linear term
        uu = self.grid.inverse(fu)
        hh = self.grid.inverse(fh)

        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))

        hx = self.grid.deriv(fh, self.grid.kx) # i k_i fh_i

        u_ux = self.grid.forward(uu*ux)

        u_h_hb_x = self.grid.deriv(uu*(hh-hb)) # i k_i f(uu*(hh-hb))_i

        fu = fup - (self.grid.dt/oo) * (u_ux +  g*hx)
        fh = fhp - (self.grid.dt/oo) * (u_h_hb_x)


        # de-aliasing
        fu[self.grid.zero_mode] = 0.0 
        fu[self.grid.dealias_modes] = 0.0 
        fh[self.grid.zero_mode] = 0.0 
        fh[self.grid.dealias_modes] = 0.0 

        return [fu, fh]

    def outs(self, fields, step):
        uu = self.grid.inverse(fields[0])
        np.save(f'uu_{step:04}', uu)
        hh = self.grid.inverse(fields[1])
        np.save(f'hh_{step:04}', hh)

    def balance(self, fields, step):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open('balance.dat', 'a') as output:
            print(*bal, file=output)
