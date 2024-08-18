''' 1D Kuramoto Sivashinsky equation '''

import numpy as np

from .solver import Solver
from .. import pseudo as ps

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
