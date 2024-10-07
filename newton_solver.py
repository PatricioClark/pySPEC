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

# Inintialize trust region function
if pm.hook:
    hookstep = mod.hookstep_function(inc_proj_X, evolve, translation, pm)
elif pm.ls:
    linesearch = mod.line_search_function(inc_proj_X, evolve, translation, pm)


#make directories for output and prints:
mod.mkdirs()


if pm.restart:
    #if restarting, load T, sx from solver.txt
    T, sx = mod.get_orb_data(pm.restart)
    # Continue solver with last newton iteration
    f_name = f'fields_{pm.restart}'
    fields = np.load(f'output/{f_name}.npz')
else:
    #if starting new run
    T, t = mod.get_orb_data(pm.restart) #t: initial time. T: guessed period
    sx = pm.sx # = 0 no prior info about sx
    if t<2000: #output2 contains directory with 500<t<2000
        output = 'output2'
    else:
        output = 'output3'
# Feed initial aproximation extracted with Kolmogorov solver
    path = '/share/data4/jcullen/pySPEC/run1/'
    idx_guess = round(t*round(1/pm.dt))
    f_name = f'fields_{idx_guess:07}'
    fields = np.load(f'{path}{output}/{f_name}.npz')


uu = fields['uu']
vv = fields['vv']

#Define variables for Newton method
X = mod.flatten_fields(uu, vv, pm, grid)
if not pm.restart:
    Y = evolve(X, T, save = True, i_newt = 0) #save evol and balance of initial candidate
else:
    Y = evolve(X, T)

Y = translation(Y,sx)
b = X-Y
b = np.append(b, [0.,0.])#RHS of linear system
print(len(b))
b_norm = np.linalg.norm(b)

# Initialize print files
if not pm.restart:
    with open('prints/solver.txt', 'x') as file1, open('prints/b_error.txt', 'x') as file2:
        file1.write('Iter newt, T, sx, |X|\n')
        file2.write('Iter newt,|b|\n')

    mod.save_X(X, '0', pm, grid)
    mod.save_X(Y, '0_T', pm, grid)


### NEWTON-GMRES ###

for i_newt in range(pm.restart+1, pm.N_newt):    

    #write to txts
    mod.write_prints(i_newt, b_norm, X, sx, T)
    
    #calculate A matrix for newton iteration
    apply_A = mod.application_function(evolve, inf_trans, translation, pm, X, T, Y, sx, i_newt)

    if pm.hook:
        #if hookstep GMRes solution must be modified with trust region
        H, beta, Q, k = GMRES(apply_A, b, i_newt, pm) #Perform GMRes iteration with A and RHS b
        X, Y, sx, T, b = hookstep(H, beta, Q, k, X, sx, T, b, i_newt)
        evolve(X, T, save = True, i_newt = i_newt) # save evol and balance of Newton iter        

    elif pm.ls:
        #if linesearch GMRes solution must be modified with trust region
        H, beta, Q, k = GMRES(apply_A, b, i_newt, pm) #Perform GMRes iteration with A and RHS b
        X, Y, sx, T, b = linesearch(H, beta, Q, k, X, sx, T, b, i_newt)
        evolve(X, T, save = True, i_newt = i_newt) # save evol and balance of Newton iter        

    else:
        #if no trust region method is applied
        dX,dsx,dT = GMRES(apply_A, b, i_newt, pm) #Perform GMRes iteration with A and RHS b
        dX = inc_proj_X(dX)
        X += dX
        T += dT
        sx += dsx
        Y = evolve(X, T, save = True, i_newt = i_newt) # save evol and balance of Newton iter
        Y = translation(Y,sx)
        b = X-Y
        b = np.append(b, [0.,0.])

    b_norm = np.linalg.norm(b)

    #For every newton step save fields at t and t+T
    mod.save_X(X, f'{i_newt}', pm, grid)
    mod.save_X(Y, f'{i_newt}_T', pm, grid)

    # no breaking to see if error reaches macheps
    # if b_norm < pm.tol_newt:
    #     break

### END OF NEWTON-GMRES ###

with open('prints/b_error.txt', 'a') as file1, open('prints/solver.txt', 'a') as file2:
    file1.write(f'{i_newt},{round(b_norm,3)}\n')
    file2.write(f'{i_newt},{round(T, 5)},{round(np.linalg.norm(X), 3)}\n')