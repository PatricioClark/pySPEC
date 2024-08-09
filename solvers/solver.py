'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import abc

class Solver(abc.ABC):
    @abc.abstractmethod
    def evolve(self, fields, T, bstep=None, sstep=None, fstep=None):
        return []

    def write_balance(self, fields, step):
        pass

    def write_spectra(self, fields, step):
        pass

    def write_fields(self, fields, step):
        pass

    def write_outputs(self, fields, step, bstep, sstep, fstep):
        if bstep is not None and step%bstep==0:
            self.write_balance(fields, step)

        if sstep is not None and step%sstep==0:
            self.write_spectra(fields, step)
            
        if fstep is not None and step%fstep==0:
            self.write_fields(fields, step)
