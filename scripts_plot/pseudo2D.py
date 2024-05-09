'''
Pseudo-spectral solver for the 2D Navier-Stokes equations

A variable-order RK scheme is used for time integrationa and the 2/3 rule is
used for dealiasing.

See Eq. (6.143) in Pope's Turbulent flows for details.

Kolmogorov forcing.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

import params2D as pm
import mod2D as mod

grid = mod.Grid(pm)

# Initial conditions
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
fu = mod.forward(uu)
fv = mod.forward(vv)
fu, fv = mod.inc_proj(fu, fv, grid)

# Forcing
kf = 4
fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
fx = mod.forward(fx)
fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
fx, fy = mod.inc_proj(fx, fy, grid)

for step in range(1, pm.Nt):

    # Store previous time step
    fup = np.copy(fu)
    fvp = np.copy(fv)

    # Print global quantities
    if step%pm.print_freq==0:
        mod.balance(fu, fv, fx, fy, grid, pm, step)

    # Save fields
    if step%pm.fields_freq==0:
        mod.save_fields(fu, fv, step)

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
