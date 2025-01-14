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
            pm: Newton parameters dictionary
            solver: solver object
        '''
        self.pm = pm
        self.solver = solver
        self.grid = solver.grid # Podría no usar un grid

    # @abc.abstractmethod
    # def F(self, X):
    #     pass

    # @abc.abstractmethod
    # def jac(self, F, X):
    #     pass

    def flatten(self, fields):
        '''Flattens fields for Newton-GMRes solver'''
        return np.concatenate([f.flatten() for f in fields])

    def unflatten(self, U):
        '''Unflatten fields'''
        ll = len(U)//self.solver.num_fields
        fields = [U[i*ll:(i+1)*ll] for i in range(self.solver.num_fields)]
        fields = [f.reshape(self.grid.shape) for f in fields]
        return fields

    # @abc.abstractmethod
    # def update_A(self, *args):
    #     pass

    # @abc.abstractmethod
    # def iterate(self, X):
    #     pass
    