'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import numpy as np
import mod_ps2D as mod

def kolmogorov_flow(grid, pm):
    '''
    Kolmogorov flow: 2D Navier-Stokes with fx = sin(2*pi*kf*y/Ly) forcing.

    See Eq. (6.143) in Pope's Turbulent flows for details on the Fourier
    decomposition of the NS equations and the pressure proyector.
    '''
    # Forcing
    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Ly)
    fx = mod.forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = mod.inc_proj(fx, fy, grid)

    def evolve(fields, T):
        ''' Evolves velocity fields to time T '''
        fu, fv = fields

        Nt = round(T/pm.dt)
        for step in range(Nt):
            # Store previous time step
            fup = np.copy(fu)
            fvp = np.copy(fv)
       
            # Time integration
            for oo in range(pm.rkord, 0, -1):
                # Non-linear term
                uu = mod.inverse(fu)
                vv = mod.inverse(fv)
                
                ux = mod.inverse(mod.deriv(fu, grid.kx))
                uy = mod.inverse(mod.deriv(fu, grid.ky))

                vx = mod.inverse(mod.deriv(fv, grid.kx))
                vy = mod.inverse(mod.deriv(fv, grid.ky))

                gx = mod.forward(uu*ux + vv*uy)
                gy = mod.forward(uu*vx + vv*vy)
                gx, gy = mod.inc_proj(gx, gy, grid)

                # Equations
                fu = fup + (grid.dt/oo) * (
                    - gx
                    - pm.nu * grid.k2 * fu 
                    + fx
                    )

                fv = fvp + (grid.dt/oo) * (
                    - gy
                    - pm.nu * grid.k2 * fv 
                    + fy
                    )

                # de-aliasing
                fu[grid.zero_mode] = 0.0 
                fv[grid.zero_mode] = 0.0 
                fu[grid.dealias_modes] = 0.0 
                fv[grid.dealias_modes] = 0.0

        return [fu, fv]
    return evolve
