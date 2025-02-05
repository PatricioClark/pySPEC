''' 2D Rayleigh-Benard solver '''

import numpy as np
import subprocess

from .solver import Solver
from .. import pseudo as ps

class SPECTER(Solver):
    '''
    SPECTER 2D flows solver. Solves Rayleigh-Benard if solver='BOUSS'
    '''

    num_fields = 3
    dim_fields = 2

    def __init__(self, pm, solver = 'BOUSS', ftypes = ['vx', 'vz', 'th']):
        super().__init__(pm)
        self.grid = ps.Grid2D_semi(pm)
        self.solver = solver
        self.ftypes = ftypes

    def evolve(self, fields, T, bstep=None, ostep=None, sstep=None, bpath='', opath='', spath=''):
        '''Evolves fields in T time and translates by sx. Calls Fortran'''
        self.write_fields(fields)

        if bstep is None:
            self.ch_params(T) #change period to evolve
        else:
            self.ch_params(T, bstep, ostep, opath) #save fields every ostep, and bal every bstep

        #run specter
        subprocess.run(f'mpirun -n {self.pm.nprocs} ./{self.solver}', shell = True)

        #save balance prints
        if bstep is not None:
            txts = 'balance.txt helicity.txt scalar.txt noslip_diagnostic.txt scalar_constant_diagnostic.txt'
            subprocess.run(f'mv {txts} {bpath}/.', shell = True)

        #load evolved fields
        fields = self.load_fields()
        return fields

    def save_binary_file(self, path, data):
        '''writes fortran file'''
        data = data.astype(np.float64).reshape(data.size,order='F')
        data.tofile(path)

    def write_fields(self, fields, idx=2, path='bin_tmp'):
        ''' Writes fields to binary file. Saves temporal fields in bin_tmp with idx=2'''

        for field, ftype in zip(fields, self.ftypes):
            self.save_binary_file(f'{path}/{ftype}.{idx:0{self.pm.ext}}.out', field)

        # Save additional empty fields required by solver
        empty_field = np.zeros(self.grid.shape, dtype=np.float64)
        for ftype in ('vy', 'pr'):
            self.save_binary_file(f'{path}/{ftype}.{idx:0{self.pm.ext}}.out', empty_field)

    def load_fields(self, path = 'bin_tmp', idx = 1): 
        '''Loads binary fields. idx = 1 for default read in bin_temp '''
        fields = []
        for ftype in self.ftypes:
            file = f'{path}/{ftype}.{idx:0{self.pm.ext}}.out'
            fields.append(np.fromfile(file,dtype=np.float64).reshape(self.grid.shape,order='F'))
        return fields

    def get_nu_kappa(self, gamma = 1):
        '''Calculates nu and kappa from ra. Assumes gamma = 1, h = 3.14159'''
        '''If different gammas are used, formula should be changed (nu neq kappa)'''
        nu = np.sqrt(gamma**2 * self.pm.Lz**4 / self.pm.ra)
        return nu, nu

    def ch_params(self, T, bstep = 0, ostep=0, opath = ''):
        '''Changes parameter.inp to update T, and sx '''
        with open('parameter.inp', 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.startswith('T_guess'):#modify period:
                lines[i] = f'T_guess = {round(T, 7)} !Initial guess for period\n'
            if line.startswith('cstep'): #modify cstep (bstep in current code)
                lines[i] = f'cstep = {bstep} !steps between writing global quantities\n'
            if line.startswith('tstep'): #modify tstep (ostep in current code)
                lines[i] = f'tstep = {ostep} !steps between saving fields\n'
            if line.startswith('odir_newt'): #modify odir_newt
                lines[i] = f'odir_newt = "{opath}" !output for saved fields\n'
            if line.startswith('dt'): #modifies dt (does not change throughout algorithm)
                lines[i] = f'dt = {self.pm.dt}   ! time step\n'

            nu, kappa = self.get_nu_kappa()
            if line.startswith('nu'): #modifies ra (does not change throughout algorithm)
                lines[i] = f'nu = {nu:.3e}         ! kinematic viscosity\n'
            if line.startswith('kappa'): #modifies ra (does not change throughout algorithm)
                lines[i] = f'kappa = {kappa:.4e}      ! scalar difussivity\n'

        #write
        with open('parameter.inp', 'w') as file:
            file.writelines(lines)

    def inc_proj(self, fields): #TODO
        ''' Solenoidal projection of fields by using Fortran subroutines''' 
        self.write_fields(fields)

        #run specter
        subprocess.run(f'mpirun -n {self.pm.nprocs} ./{self.solver}_PROJ', shell = True)

        #load evolved fields
        fields = self.load_fields()
        return fields
