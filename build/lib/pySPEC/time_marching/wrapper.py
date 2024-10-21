'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import abc
import numpy as np

from .solver import Solver
from .. import pseudo as ps

class Wrapper(Solver, abc.ABC):
    ''' Abstract Wrapper solver class '''
    def __init__(self, pm):
        super().__init__(pm)

    def evolve(self, fields, T, bstep=None, sstep=None, ostep=None):
        ''' Evolves velocity fields to time T  by calling GHOST/SPECTER'''

        return []
