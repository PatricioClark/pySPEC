import abc
import numpy as np

from .. import pseudo as ps

class Solver(abc.ABC):
    ''' Abstract solver class

    All solvers must implement the following methods:
        - evolve: Evolve solver method
        - balance: Energy balance
        - spectra: Energy spectra
        - outs: Outputs

    All solvers have defined the following: 
        - num_fields: Number of fields
        - dim_fields: Dimension of the fields
    '''
    num_fields: int
    dim_fields: int

    def __init__(self, pm):
        ''' Initializes the solver

        Parameters:
        ----------
            pm: parameter dictionary
        '''
        self.pm = pm
        self.pm.dim = self.dim_fields
        self.grid: ps.Grid1D | ps.Grid2D

    @abc.abstractmethod
    def evolve(self, fields, T, bstep=None, sstep=None, ostep=None):
        return []

    def balance(self, fields, step):
        return []

    def spectra(self, fields, step):
        pass

    def outs(self, fields, step):
        pass

    def write_outputs(self, fields, step, bstep, sstep, ostep):
        if bstep is not None and step%bstep==0:
            bal = self.balance(fields, step)
            with open('balance.dat', 'a') as output:
                print(*bal, file=output)

        if sstep is not None and step%sstep==0:
            self.spectra(fields, step)
            
        if ostep is not None and step%ostep==0:
            self.outs(fields, step)
