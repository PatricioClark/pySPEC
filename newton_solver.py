import numpy as np
import matplotlib.pyplot as plt
from gmres_kol import GMRES
import os

import params2D as pm
import mod2D as mod

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)
inc_proj_X = mod.inc_proj_X_function(grid, pm)
inf_trans = mod.inf_trans_function(grid, pm)
translation = mod.translation_function(grid, pm)
hookstep = mod.hookstep_function(inc_proj_X, evolve, translation, pm)

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

#make directories for output and prints:
mkdir('output')
mkdir('prints')
for print_dir in ('error_gmres', 'hookstep', 'apply_A'):
    mkdir(f'prints/{print_dir}')

### Feed initial aproximation extracted with Kolmogorov solver
path = '/share/data4/jcullen/pySPEC/run1/'
idx_guess = int(pm.t_guess*int(1/pm.dt))
f_name = f'fields_{idx_guess:07}'
fields = np.load(f'{path}output2/{f_name}.npz')
uu = fields['uu']
vv = fields['vv']

X = mod.flatten_fields(uu, vv, pm, grid)
T = pm.T_guess
sx = pm.sx_guess
Y = evolve(X, T)
Y = translation(Y,sx)
b = X-Y
b = np.append(b, [0.,0.])
print(len(b))
b_norm = np.linalg.norm(b)
with open('prints/solver.txt', 'w') as file1, open('prints/b_error.txt', 'w') as file2:
    file1.write('Iter newt, T, sx, |X|\n')
    file2.write('Iter newt,|b|\n')

mod.save_X(X, '0', pm, grid)
mod.save_X(Y, '0_T', pm, grid)

for i_newt in range(1, pm.N_newt):    

    with open('prints/b_error.txt', 'a') as file1, open('prints/solver.txt', 'a') as file2, \
    open(f'prints/error_gmres/iter{i_newt}.txt', 'w') as file3, open(f'prints/hookstep/iter{i_newt}.txt', 'w') as file4,\
        open(f'prints/apply_A/iter{i_newt}.txt', 'w') as file5:
        file1.write(f'{i_newt-1},{round(b_norm,3)}\n')
        file2.write(f'{i_newt-1},{round(T, 8)},{round(sx, 8)},{round(np.linalg.norm(X), 3)}\n')
        file3.write('iter gmres, error\n')
        file4.write('iter hookstep, |b|\n')
        file5.write('|dX|,|dY/dX|,|dX/dt|,|dY/dT|,t_proj,Tx_proj\n')

    apply_A = mod.application_function(evolve, inf_trans, translation, pm, X, T, Y, sx, i_newt)
    if not pm.hook:
        dX,dsx,dT = GMRES(apply_A, b, pm.N_gmres, i_newt, tol = pm.tol_gmres)
        dX = inc_proj_X(dX)
        X += dX
        T += dT
        sx += dsx
        Y = evolve(X, T)
        Y = translation(Y,sx)
        b = X-Y
        b = np.append(b, [0.,0.])
    else:
        H, beta, Q, k = GMRES(apply_A, b, pm.N_gmres, i_newt, pm.tol_gmres, pm.hook)
        X, Y, sx, T, b = hookstep(H, beta, Q, k, X, sx, T, b, i_newt)
    b_norm = np.linalg.norm(b)

    #Save fields
    mod.save_X(X, f'{i_newt}', pm, grid)
    mod.save_X(Y, f'{i_newt}_T', pm, grid)

    if b_norm < pm.tol_newt:
        break

with open('prints/b_error.txt', 'a') as file1, open('prints/solver.txt', 'a') as file2:
    file1.write(f'{i_newt},{round(b_norm,3)}\n')
    file2.write(f'{i_newt},{round(T, 5)},{round(np.linalg.norm(X), 3)}\n')


# TODO: usar numba jit