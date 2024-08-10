'''
Collection of different solvers.

They must all return the fields in Fourier space.
'''

import abc

class Solver(abc.ABC):
    @abc.abstractmethod
    def evolve(self, fields, T, bstep=None, sstep=None, fstep=None):
        return []

    def balance(self, fields, step):
        return []

    def spectra(self, fields, step):
        pass

    def fields(self, fields, step):
        pass

    def write_outputs(self, fields, step, bstep, sstep, fstep):
        if bstep is not None and step%bstep==0:
            bal = self.balance(fields, step)
            with open('balance.dat', 'a') as output:
                print(*bal, file=output)

        if sstep is not None and step%sstep==0:
            self.spectra(fields, step)
            
        if fstep is not None and step%fstep==0:
            self.fields(fields, step)
