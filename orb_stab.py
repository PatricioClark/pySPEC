import numpy as np
import matplotlib.pyplot as plt
from gmres_kol import arnoldi_eig
import params2D as pm
import mod2D as mod

#Compute converged orbs eigenvalues (Floquet Multipliers) using Arnoldi iteration

# Initialize solver
grid = mod.Grid(pm)
evolve  = mod.evolution_function(grid, pm)
translation = mod.translation_function(grid, pm)


def application_function(X,Y,s, T):
    norm_X = np.linalg.norm(X)
    def apply_J(dX):
        ''' Applies J (jacobian of poincare map) matrix to vector (dX)^t  '''        
        norm_dX = np.linalg.norm(dX)
        epsilon = 1e-7*norm_X/norm_dX

        deriv_x0 = evolve(X+epsilon*dX, T)
        deriv_x0 = translation(deriv_x0,s) - Y
        deriv_x0 = deriv_x0/epsilon
        return deriv_x0

    return apply_J

mod.mkdir('eigvals')

#directories of converged orbits
orb_dirs = ['orb05dt']

for orb in orb_dirs:
    b_error = np.loadtxt(f'../{orb}/prints/b_error.txt', delimiter = ',', skiprows = 1, ndmin = 2)
    #find the iteration which minimizes |b|
    i_min = b_error[:,1].argmin()
    b_min = b_error[i_min,1]
    solver = np.loadtxt(f'../{orb}/prints/solver.txt', delimiter = ',', skiprows = 1, ndmin = 2)
    T = solver[i_min, 1]
    s = solver[i_min, 2]
    n_min = int(b_error[i_min, 0])

    #load X
    f_name = f'fields_{n_min}'
    fields = np.load(f'../{orb}/output/{f_name}.npz')
    uu = fields['uu']
    vv = fields['vv']
    X = mod.flatten_fields(uu, vv, pm, grid)

    #load Y
    fields = np.load(f'../{orb}/output/{f_name}_T.npz')
    uu = fields['uu']
    vv = fields['vv']
    Y = mod.flatten_fields(uu, vv, pm, grid)

    #load application function for J
    apply_J = application_function(X, Y, s, T)

    #find eig of J
    #define b by adding noise to initial converged state (Viswanath)
    # b = X - Y# + np.random.randn(len(X))*np.linalg.norm(X)*1e-5
    b = np.random.randn(len(X))

    eigvals, eigvecs, Q = arnoldi_eig(apply_J, b, pm, orb)

    with open(f'eigvals/{orb}.txt', 'a') as file:
        file.write('eigval nmb, re, im \n')
        for i,eigval in enumerate(eigvals):
            file.write(f'{i+1},{eigval.real},{eigval.imag}\n')

    np.save(f'eigvals/Q_{orb}.npy', Q)
