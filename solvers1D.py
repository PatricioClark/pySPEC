'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import numpy as np
import mod_ps1D as mod

def KuramotoSivashinsky(grid, pm):
    def evolve(fields, T):
        ''' Evolves velocity fields to time T '''
        fu = fields[0]

        Nt = round(T/pm.dt)
        for step in range(Nt):
            # Store previous time step
            fup = np.copy(fu)

            # Time integration
            for oo in range(pm.rkord, 0, -1):
                # Non-linear term
                uu  = mod.inverse(fu)
                fu2 = mod.forward(uu**2)

                fu = fup + (grid.dt/oo) * (
                    - (0.5*1.0j*grid.kx*fu2)
                    + ((grid.k2)*fu) 
                    - ((grid.k2**2)*fu)
                    )

                # de-aliasing
                fu[grid.zero_mode] = 0.0 
                fu[grid.dealias_modes] = 0.0 

        return [fu]
    return evolve
