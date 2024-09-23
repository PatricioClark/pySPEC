'''
Newton Solver Definition
'''

import abc
import numpy as np

from .. import pseudo as ps

class NewtonSolver(abc.ABC):

    def __init__(self, pm, solver):
        ''' Initializes the Newton solver

        Parameters:
        ----------
            pm: parameter dictionary
            solver: solver object
        '''
        self.pm = pm
        self.solver = solver
        self.grid = solver.grid

    def iterate(self, fields, trust_region='hook'):
        for i_newt in range(self.pm.restart+1, self.pm.N_newt):    

            # Write to txts
            # mod.write_prints(i_newt, b_norm, X, sx, T)
            
            # Calculate A matrix for newton iteration
            apply_A = mod.application_function(solver.evolve,
                                               inf_trans,
                                               translation,
                                               pm, X, T, Y, sx,
                                               i_newt)
            self.update_A()

            # Perform GMRes iteration with A and RHS b
            H, beta, Q, k = GMRES(self.apply_A, b, i_newt, self.pm)

            if trust_region == 'hook':
                X, Y, sx, T, b = hookstep(H, beta, Q, k, X, sx, T, b, i_newt)
            elif trust_region == 'linesearch':
                X, Y, sx, T, b = linesearch(H, beta, Q, k, X, sx, T, b, i_newt)
            else:
                pass

            b_norm = np.linalg.norm(b)

            #For every newton step save fields at t and t+T
            mod.save_X(X, f'{i_newt}', pm, grid)
            mod.save_X(Y, f'{i_newt}_T', pm, grid)

            if b_norm < self.pm.tol_newt:
                break

    @abc.abstractmethod
    def update_A(self):
        pass
