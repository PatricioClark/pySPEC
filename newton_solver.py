import numpy as np
import matplotlib.pyplot as plt
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
with open('prints/solver.txt', 'w') as file1, open('prints/b_error.txt', 'w') as file2:
    file1.write('Iter newt, T, |X|\n')
    file2.write('Iter newt,|b|\n')

mod.save_X(X, '0', pm)
mod.save_X(Y, '0_T', pm)

for i_newt in range(1, pm.N_newt):    

    with open('prints/b_error.txt', 'a') as file1, open('prints/solver.txt', 'a') as file2, \
    open(f'prints/error_gmres/iter{i_newt}.txt', 'w') as file3, open(f'prints/hookstep/iter{i_newt}.txt', 'w') as file4,\
        open(f'prints/apply_A/iter{i_newt}.txt', 'w') as file5:
        file1.write(f'{i_newt-1},{round(b_norm,3)}\n')
        file2.write(f'{i_newt-1},{round(T, 5)},{round(np.linalg.norm(X), 3)}\n')
        file3.write('iter gmres, error\n')
        file4.write('iter hookstep, |b|\n')
        file5.write('|dX|,|dY/dX|,|dX/dt|,|dY/dT|,t_proj\n')

    apply_A = mod.application_function(evolve, pm, X, T, Y, i_newt)
    if not pm.hook:
        dX,dT = GMRES(apply_A, b, pm.N_gmres, i_newt ,tol = pm.tol_gmres)
        dX = inc_proj_X(dX)
        X += dX
        T += dT
        Y = evolve(X, T)
        b = X-Y
        b = np.append(b, 0.)
    else:
        H, beta, Q, k = GMRES(apply_A, b, pm.N_gmres, i_newt, tol = pm.tol_gmres, hookstep = pm.hook)
        X, Y, T, b = mod.hookstep(H, beta, Q, k, X, T, b, inc_proj_X, evolve, pm, i_newt)
    b_norm = np.linalg.norm(b)

    #Save fields
    mod.save_X(X, f'{i_newt}', pm)
    mod.save_X(Y, f'{i_newt}_T', pm)

    if b_norm < pm.tol_newt:
        break

with open('prints/b_error.txt', 'a') as file1, open('prints/solver.txt', 'a') as file2:
    file1.write(f'{i_newt},{round(b_norm,3)}\n')
    file2.write(f'{i_newt},{round(T, 5)},{round(np.linalg.norm(X), 3)}\n')


# TODO: usar numba jit
# TODO: agregar shift en x
# TODO: agregar Hookstep
