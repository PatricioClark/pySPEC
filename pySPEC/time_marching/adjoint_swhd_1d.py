''' 1D Adjoint Shallow Water Equations '''

import numpy as np

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class Adjoint_SWHD_1D(PseudoSpectral):
    ''' 1D Adjoint Shallow Water Equations
        ut_ + u ux_ + (h-hb) hx_ = 2(u-um)
        ht_ + u hx_ + g ux_  = 2(h-hm)
    where u,h are physical velocity and height fields,
    u_,h_ the adjoint state fields,
    um,hm the physical field measurements,
    and hb is bottom topography condition.
    '''

    num_fields = 7 # adoint fields u_ , h_ ; physical fields u, h ; measurements um, hm ; contour hb
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)

        # self.hb = np.zeros_like(self.grid.xx) # test flat bottom

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu_  = fields[0]
        fu_p = prev[0]

        fh_  = fields[1]
        fh_p = prev[1]

        fu  = fields[2]
        fup = prev[2]

        fh  = fields[3]
        fhp = prev[3]

        fum = fields[4] # u measurments
        fhm = fields[5] # h measurements
        fhb = fields[6] # bottom contour

        # Non-linear term
        uu_ = self.grid.inverse(fu_)
        hh_ = self.grid.inverse(fh_)
        uu = self.grid.inverse(fu)
        hh = self.grid.inverse(fh)
        hb = self.grid.inverse(fhb)

        fux_ = self.grid.deriv(fu_, self.grid.kx)
        ux_ = self.grid.inverse(fux_)
        fhx_ = self.grid.deriv(fh_, self.grid.kx)
        hx_ = self.grid.inverse(fhx_)

        fu_ux_ = self.grid.forward(uu*ux_)
        fu_hx_ = self.grid.forward(uu*hx_)

        fh_hb_hx_ = self.grid.forward((hh-hb)*hx_)

        # backwards integration in time
        fu_ = fu_p - (self.grid.dt/oo) * (2*(fu-fum) - fu_ux_ - fh_hb_hx_)
        fh_ = fh_p - (self.grid.dt/oo) * (2*(fh-fhm) - self.pm.g*fux_ - fu_hx_)

        # de-aliasing
        # fu[self.grid.zero_mode] = 0.0
        fu[self.grid.dealias_modes] = 0.0
        # fh[self.grid.zero_mode] = 0.0 # zero mode for h is not null
        fh[self.grid.dealias_modes] = 0.0

        return [fu_,fh_, fu, fh, fum, fhm, fhb]

    def outs(self, fields, step):
        uu_ = self.grid.inverse(fields[0])
        np.save(f'{self.pm.out_path}/adjuu_{step:04}', uu_)
        hh_ = self.grid.inverse(fields[1])
        np.save(f'{self.pm.out_path}/adjhh_{step:04}', hh_)


    def balance(self, fields, step):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)
