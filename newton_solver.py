import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from gmres_kol import GMRES

import params2D as pm
import mod2D as mod

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)
inc_proj_X = mod.inc_proj_X_function(grid, pm)

### Feed initial aproximation extracted with Kolmogorov solver
path = '/share/data4/jcullen/pySPEC/run1/'
idx_guess = int(pm.t_guess/(pm.save_freq*pm.dt))
f_name = f'fields_{idx_guess:06}'
fields = np.load(f'{path}output/{f_name}.npz')
uu = fields['uu']
vv = fields['vv']

X = mod.flatten_fields(uu, vv)
T = pm.T_guess
Y = evolve(X, T)
b = X-Y
b = np.append(b, 0.)
b_norm = np.linalg.norm(b)
with open('solver.txt', 'w') as file1, open('b_error.txt', 'w') as file2:
    file1.write(f'Initial T= {round(T, 3)}. Initial |X|={round(np.linalg.norm(X), 3)} \n')
    file2.write(f'Initial |b| = {round(b_norm,3)}\n')
with open('error_gmres.txt', 'w') as file1, open('hookstep.txt', 'w') as file2:
    file1.write('')
    file2.write('')

mod.save_fields(X, '0', pm)
mod.save_fields(Y, '0_T', pm)

for i_newt in range(pm.N_newt):    
    apply_A = mod.application_function(evolve, pm, X, T, Y)
    if pm.hook == False:
        dX,dT = GMRES(apply_A, b, pm.N_gmres, pm.tol_gmres)
        dX = inc_proj_X(dX)
        X += dX
        T += dT
        Y = evolve(X, T)
        b = X-Y
        b = np.append(b, 0.)
    else:
        H, beta, Q, k = GMRES(apply_A, b, pm.N_gmres, pm.tol_gmres, pm.hook)
        X, Y, T, b = mod.hookstep(H, beta, Q, k, X, T, b, inc_proj_X, evolve, pm)
    b_norm = np.linalg.norm(b)
    with open('b_error.txt', 'a') as file:
        file.write(f'Iter. Newton: {i_newt+1}. |b| = {round(b_norm,3)}\n')
    with open('solver.txt', 'a') as file1, open('hookstep.txt', 'a') as file2:
        file1.write(f'Iter. Newton: {i_newt+1}. T= {T}. |X|={round(np.linalg.norm(X), 3)} \n')
        file2.write(f'Iter. Newton: {i_newt+1}\n')
    #Save fields
    mod.save_fields(X, f'{i_newt+1}', pm)
    mod.save_fields(Y, f'{i_newt+1}_T', pm)
    if b_norm < pm.tol_newt:
        break

# TODO: usar numba jit
# TODO: agregar shift en x
# TODO: agregar Hookstep
