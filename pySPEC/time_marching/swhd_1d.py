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
        self.iit = pm.iit
        self.hb_path = pm.hb_path
        self.hb = np.load(f'{self.hb_path}/hb_{self.iit}.npy') # hb field at current GD iteration

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu  = fields[0]
        fup = prev[0]

        fh  = fields[1]
        fhp = prev[1]

        hb = self.hb
        # save hb stats to debug
        # save hb stats to debug
        # if self.current_step == 2:
        #         loss = [f'{self.iit}', f'{hb.mean():.6e}'  , f'{hb.std():.6e}' ,  f'{hb.max():.6e}' , f'{hb.min():.6e}']
        #         with open(f'{self.pm.hb_path}/front_hb_stats.dat', 'a') as output:
        #             print(*loss, file=output)

        # Non-linear term
        uu = self.grid.inverse(fu)
        hh = self.grid.inverse(fh)

        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))

        fhx = self.grid.deriv(fh, self.grid.kx) # i k_i fh_i

        fu_ux = self.grid.forward(uu*ux)

        fu_h_hb_x = self.grid.deriv(self.grid.forward(uu*(hh-hb)) , self.grid.kx) # i k_i f(uu*(hh-hb))_i

        fu = fup + (self.grid.dt/oo) * (-fu_ux -  self.pm.g*fhx)
        fh = fhp + (self.grid.dt/oo) * (-fu_h_hb_x)

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0
        fu[self.grid.dealias_modes] = 0.0
        # fh[self.grid.zero_mode] = 0.0 # zero mode for h is not null
        fh[self.grid.dealias_modes] = 0.0

        return [fu, fh]

    def outs(self, fields, step):
        uu = self.grid.inverse(fields[0])
        np.save(f'{self.pm.out_path}/uu_{step:04}', uu)
        hh = self.grid.inverse(fields[1])
        np.save(f'{self.pm.out_path}/hh_{step:04}', hh)

    def balance(self, fields, step):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)
