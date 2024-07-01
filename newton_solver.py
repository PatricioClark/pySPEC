import numpy as np
import matplotlib.pyplot as plt
from gmres_kol import GMRES
import params2D as pm
import mod2D as mod

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)
inc_proj_X = mod.inc_proj_X_function(grid, pm)
inf_trans = mod.inf_trans_function(grid, pm)
translation = mod.translation_function(grid, pm)
hookstep = mod.hookstep_function(inc_proj_X, evolve, translation, pm)


#make directories for output and prints:
mod.mkdir('output')
mod.mkdir('prints')
for print_dir in ('error_gmres', 'hookstep', 'apply_A'):
    mod.mkdir(f'prints/{print_dir}')

if pm.restart:
### Continue solver with last newton iteration
    f_name = f'fields_{pm.restart}'
    fields = np.load(f'output/{f_name}.npz')
else:
    if pm.t_guess<2000: #output2 contains directory with 500<t<2000
        output = 'output2'
    else:
        output = 'output3'
### Feed initial aproximation extracted with Kolmogorov solver
    path = '/share/data4/jcullen/pySPEC/run1/'
    idx_guess = int(pm.t_guess*int(1/pm.dt))
    f_name = f'fields_{idx_guess:07}'
    fields = np.load(f'{path}{output}/{f_name}.npz')

uu = fields['uu']
vv = fields['vv']

X = mod.flatten_fields(uu, vv, pm, grid)
if not pm.restart:
    T = pm.T_guess
    sx = pm.sx_guess
else:
    #if restarting, load T, sx from solver.txt
    solver = np.loadtxt('prints/solver.txt', delimiter = ',', skiprows = 1)
    T = solver[pm.restart,1]
    sx = solver[pm.restart,2]

Y = evolve(X, T)
Y = translation(Y,sx)
b = X-Y
b = np.append(b, [0.,0.])#RHS of linear system
print(len(b))
b_norm = np.linalg.norm(b)


if not pm.restart:
    with open('prints/solver.txt', 'w') as file1, open('prints/b_error.txt', 'w') as file2:
        file1.write('Iter newt, T, sx, |X|\n')
        file2.write('Iter newt,|b|\n')

    mod.save_X(X, '0', pm, grid)
    mod.save_X(Y, '0_T', pm, grid)

for i_newt in range(pm.restart+1, pm.N_newt):    

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
        dX,dsx,dT = GMRES(apply_A, b, i_newt, pm) #Perform GMRes iteration with A and RHS b
        dX = inc_proj_X(dX)
        X += dX
        T += dT
        sx += dsx
        Y = evolve(X, T)
        Y = translation(Y,sx)
        b = X-Y
        b = np.append(b, [0.,0.])
    else:
        #if hookstep GMRes solution must be modified with trust region
        H, beta, Q, k = GMRES(apply_A, b, i_newt, pm) #Perform GMRes iteration with A and RHS b
        X, Y, sx, T, b = hookstep(H, beta, Q, k, X, sx, T, b, i_newt)
    b_norm = np.linalg.norm(b)

    #For every newton step save fields at t and t+T
    mod.save_X(X, f'{i_newt}', pm, grid)
    mod.save_X(Y, f'{i_newt}_T', pm, grid)

    if b_norm < pm.tol_newt:
        break

with open('prints/b_error.txt', 'a') as file1, open('prints/solver.txt', 'a') as file2:
    file1.write(f'{i_newt},{round(b_norm,3)}\n')
    file2.write(f'{i_newt},{round(T, 5)},{round(np.linalg.norm(X), 3)}\n')