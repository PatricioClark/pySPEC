'''
Newton Solver Definition
'''

import abc
import numpy as np

from .. import pseudo as ps

class NewtonSolver(abc.ABC):

    def __init__(self, pmN, solver):
        ''' Initializes the Newton solver

        Parameters:
        ----------
            pmN: Newton parameters dictionary
            solver: solver object
        '''
        self.pmN = pmN
        self.solver = solver
        # self.grid = solver.grid # Podría no usar un grid

    # @abc.abstractmethod
    # def F(self, X):
    #     pass

    # @abc.abstractmethod
    # def jac(self, F, X):
    #     pass

    @abc.abstractmethod
    def flatten(self, fields):
        pass

    @abc.abstractmethod
    def unflatten(self, U):
        pass

    @abc.abstractmethod
    def update_A(self, *args):
        pass

    @abc.abstractmethod
    def iterate(self, X):
        pass
    
    # def iterate(self, X):
    #     for i_newt in range(self.pmN.restart+1, self.pmN.N_newt):    

    #         # Write to txts
    #         if self.pmN.verbose:
    #             self.write_prints(i_newt, b_norm, U, sx, T)
            
    #         # Calculate A matrix for newton iteration
    #         A = self.update_A(X)

    #         # Perform GMRes iteration with A and RHS b
    #         H, beta, Q, k = GMRES(self.apply_A, b, i_newt, self.pmN)

    #         if trust_region == 'hook':
    #             X, Y, sx, T, b = hookstep(H, beta, Q, k, X, sx, T, b, i_newt)
    #         elif trust_region == 'linesearch':
    #             X, Y, sx, T, b = linesearch(H, beta, Q, k, X, sx, T, b, i_newt)
    #         else:
    #             pass

    #         b_norm = np.linalg.norm(b)

    #         #For every newton step save fields at t and t+T
    #         mod.save_X(X, f'{i_newt}', pmN, grid)
    #         mod.save_X(Y, f'{i_newt}_T', pmN, grid)

    #         if b_norm < self.pmN.tol_newt:
    #             break

