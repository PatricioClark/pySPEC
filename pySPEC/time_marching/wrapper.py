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

    @abc.abstractmethod
    def evolve(self, fields, T, bstep=None, ostep=None, sstep=None, bpath = '', opath = '', spath = ''):
        ''' Evolves velocity fields to time T  by calling GHOST/SPECTER'''
        pass

    @abc.abstractmethod
    def write_fields(self, fields, *args):
        pass

    @abc.abstractmethod
    def ch_params(self, T, *args):
        pass