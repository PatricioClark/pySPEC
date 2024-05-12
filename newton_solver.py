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
start_time = time.time()
Y = evolve(X, T)
time_dif = round(time.time() - start_time,2)
b = X-Y
b = np.append(b, 0.)
b_norm = np.linalg.norm(b)
with open('solver.txt', 'w') as file:
    file.write(f'Finished evolving X. time diff= {time_dif} \n')
    file.write(f'Initial T= {round(T, 3)}. Initial |X|={round(np.linalg.norm(X), 3)} \n')
with open('b_error.txt', 'w') as file:
    file.write(f'Initial |b| = {round(b_norm,3)}\n')
with open('error_gmres.txt', 'w') as file:
    file.write('')

uu, vv = mod.unflatten_fields(X, pm)
np.savez(f'output/fields_0.npz', uu=uu,vv=vv)
uu, vv = mod.unflatten_fields(Y, pm)
np.savez(f'output/fields_0_T.npz', uu=uu,vv=vv)

for i_newt in range(pm.N_newt):    
    apply_A = mod.application_function(evolve, pm, X, T, Y)
    dX,dT,er = GMRES(apply_A, b, pm.N_gmres, pm.tol_gmres)
    dX = inc_proj_X(dX)
    X += dX
    T += dT
    Y = evolve(X, T)
    b = X-Y
    b = np.append(b, 0.)
    b_norm = np.linalg.norm(b)
    with open('b_error.txt', 'a') as file:
        file.write(f'Iter. Newton: {i_newt+1}. |b| = {round(b_norm,3)}\n')
    with open('solver.txt', 'a') as file:
        file.write(f'Iter. Newton: {i_newt+1}. T= {T}. |X|={round(np.linalg.norm(X), 3)} \n')
    #Save fields
    uu, vv = mod.unflatten_fields(X, pm)
    np.savez(f'output/fields_{i_newt+1}.npz', uu=uu,vv=vv)
    uu, vv = mod.unflatten_fields(Y, pm)
    np.savez(f'output/fields_{i_newt+1}_T.npz', uu=uu,vv=vv)
    if b_norm < pm.tol_newt:
        break

# TODO: usar numba jit
# TODO: agregar shift en x
# TODO: agregar Hookstep
