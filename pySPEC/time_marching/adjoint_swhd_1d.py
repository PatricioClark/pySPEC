''' 1D Adjoint Shallow Water Equations '''

import numpy as np

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class Adjoint_SWHD_1D(PseudoSpectral):
    ''' 1D Adjoint Shallow Water Equations
        ut_ + u ux_ + (u*u_)x + (h-hb) hx_ = 2(u-um)
        ht_ + u hx_ + g ux_  = 2(h-hm)
    where u,h are physical velocity and height fields,
    u_,h_ the adjoint state fields,
    um,hm the physical field measurements,
    and hb is bottom topography condition.
    '''

    num_fields = 2 # adoint fields u_ , h_
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)
        self.iit = pm.iit
        self.data_path = pm.data_path
        self.field_path = pm.field_path
        self.hb_path = pm.hb_path

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu_  = fields[0]
        fu_p = prev[0]

        fh_  = fields[1]
        fh_p = prev[1]

        # Access the step from the evolve method via the class attribute
        step = self.current_step

        # get physical fields and measurements from T to t0, back stepping in time
        Nt = round(self.pm.T/self.pm.dt)
        back_step = Nt-1 - step
        uu = np.load(f'{self.field_path}/uu_{back_step:04}.npy') # u field at current time step
        hh = np.load(f'{self.field_path}/hh_{back_step:04}.npy') # h field at current time step
        hb = np.load(f'{self.hb_path}/hb_{self.iit}.npy') # hb field at current GD iteration

        fu  = self.grid.forward(uu)
        fh  = self.grid.forward(hh)
        fhb = self.grid.forward(hb)

        fum = self.grid.forward(np.load(f'{self.data_path}/uu_{back_step:04}.npy')) # u measurments at current time step
        fhm = self.grid.forward(np.load(f'{self.data_path}/hh_{back_step:04}.npy')) # h measurements at current time step

        # calculate terms
        uu_ = self.grid.inverse(fu_)
        # hh_ = self.grid.inverse(fh_)


        fux_ = self.grid.deriv(fu_, self.grid.kx)
        ux_ = self.grid.inverse(fux_)
        fhx_ = self.grid.deriv(fh_, self.grid.kx)
        hx_ = self.grid.inverse(fhx_)

        fu_ux_ = self.grid.forward(uu*ux_)
        fu_u_x = self.grid.deriv(self.grid.forward(uu*uu_), self.grid.kx)
        fu_hx_ = self.grid.forward(uu*hx_)

        fh_hb_hx_ = self.grid.forward((hh-hb)*hx_)

        # backwards integration in time
        fu_ = fu_p - (self.grid.dt/oo) * (2*(fu-fum) - fu_ux_ - fu_u_x - fh_hb_hx_)
        fh_ = fh_p - (self.grid.dt/oo) * (2*(fh-fhm) - self.pm.g*fux_ - fu_hx_)

        # de-aliasing
        # fu[self.grid.zero_mode] = 0.0
        fu[self.grid.dealias_modes] = 0.0
        # fh[self.grid.zero_mode] = 0.0 # zero mode for h is not null
        fh[self.grid.dealias_modes] = 0.0

        return [fu_,fh_] # step and back_step for debugging

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
