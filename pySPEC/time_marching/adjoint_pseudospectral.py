import abc
import numpy as np

from .solver import Solver
from .. import pseudo as ps

class AdjPseudoSpectral(Solver, abc.ABC):
    ''' Abstract (Ajoint) Pseudospectral solver class.
    It is able to incorporate data fields at each step.
    All solvers must implement the following methods:
        - rkstep: Runge-Kutta step
        - balance: Energy balance
        - spectra: Energy spectra
        - outs: Outputs
    '''
    def __init__(self, pm):
        super().__init__(pm)

    def evolve(self, fields, T, data_path, field_path, bstep=None, sstep=None, ostep=None ):
        ''' Evolves velocity fields to time T '''

        # Forward transform
        fields = [self.grid.forward(ff) for ff in fields]
        Nt = round(T/self.pm.dt)
        for step in range(Nt):
            # Store previous time step
            prev = np.copy(fields)

            # Time integration
            for oo in range(self.pm.rkord, 0, -1):
                fields = self.rkstep(fields, prev, oo, step, data_path, field_path)

            # Write outputs
            self.write_outputs(fields, step, bstep, sstep, ostep)

        # Inverse transform
        fields = [self.grid.inverse(ff) for ff in fields]

        return fields

    @abc.abstractmethod
    def rkstep(self, fields, prev, oo):
        return []
