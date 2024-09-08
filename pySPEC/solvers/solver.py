'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import abc
import numpy as np

from .. import pseudo as ps

class Solver(abc.ABC):
    ''' Abstract solver class

    All solvers must implement the following methods:
        - rkstep: Runge-Kutta step
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
        self.grid = ps.Grid(pm)

    def evolve(self, fields, T, bstep=None, sstep=None, ostep=None):
        ''' Evolves velocity fields to time T '''

        Nt = round(T/self.pm.dt)
        for step in range(Nt):
            # Store previous time step
            prev = np.copy(fields)

            # Time integration
            for oo in range(self.pm.rkord, 0, -1):
                fields = self.rkstep(fields, prev, oo)

            # Write outputs
            self.write_outputs(fields, step, bstep, sstep, ostep)

        return fields

    @abc.abstractmethod
    def rkstep(self, fields, prev, oo):
        return []

    def balance(self, fields, step):
        eng = ps.energy(fields, self.grid)
        return [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']

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
