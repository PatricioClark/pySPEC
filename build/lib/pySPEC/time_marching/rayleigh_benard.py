''' 2D Rayleigh-Benard solver '''

import numpy as np
import subprocess

from .pseudospectral import PseudoSpectral
from .wrapper import Wrapper
from .. import pseudo as ps

class RayleighBenard(Wrapper):
    '''
    Rayleigh Benard 2D flow
    '''

    num_fields = 3
    dim_fields = 2

    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid2D_wrap(pm)    

    def evolve(self, fields, T, bstep=None, ostep=None, sstep=None, bpath='', opath='', spath=''):
        '''Evolves fields in T time and translates by sx. Calls Fortran'''
        self.write_fields(fields)

        if bstep is None:
            self.ch_params(T) #change period to evolve
        else:
            self.ch_params(T, self.pm.bstep, self.pm.ostep, opath) #save fields every ostep, and bal every bstep

        #run specter
        subprocess.run(f'mpirun -n {self.pm.nprocs} ./BOUSS', shell = True)

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
        ftypes = ['vx', 'vz', 'th']

        for field, ftype in zip(fields, ftypes):
            self.save_binary_file(f'{path}/{ftype}.{idx:0{self.pm.ext}}.out', field)

        # Save additional empty fields required by solver
        empty_field = np.zeros(self.grid.shape, dtype=np.float64)
        for ftype in ('vy', 'pr'):
            self.save_binary_file(f'{path}/{ftype}.{idx:0{self.pm.ext}}.out', empty_field)

    def load_fields(self, path = 'bin_tmp', idx = 1): 
        '''Loads binary fields. idx = 1 for default read in bin_temp '''
        ftypes = ['vx', 'vz', 'th']
        fields = []

        for ftype in ftypes:
            file = f'{path}/{ftype}.{idx:0{self.pm.ext}}.out'
            fields.append(np.fromfile(file,dtype=np.float64).reshape(self.grid.shape,order='F'))
        return fields

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

        #write
        with open('parameter.inp', 'w') as file:
            file.writelines(lines)

    def inc_proj(self, fields): #TODO
        # def sol_project(U, pr, pm, i_newt, i_gmres=0):
        ''' Solenoidal projection of fields by using Fortran subroutines''' 
        self.write_fields(fields,'bin_tmp')

        # if i_gmres:
        #     direc = f'sol_proj/i_newt{i_newt:02}/i_gmres{i_gmres:02}' 
        # else:
        #     direc = f'sol_proj/i_newt{i_newt:02}'
        # mkdir(direc)
        # save_U(U, pr, 'pre_sol_proj', pm, direc = direc)

        # with open('dU_norm.txt', 'w') as file:
        #     file.write(f'|dU1|,{np.linalg.norm(U):.4e}\n')

        #run specter
        subprocess.run(f'mpirun -n {self.pm.nprocs} ./BOUSS_PROJ', shell = True)

        #load evolved fields
        fields = self.load_fields()

        # with open('dU_norm.txt', 'a') as file:
        #     file.write(f'|dU2|,{np.linalg.norm(U):.4e}')

        #save balance prints
        # txts = 'balance.txt helicity.txt scalar.txt noslip_diagnostic.txt scalar_constant_diagnostic.txt dU_norm.txt'
        # subprocess.run(f'mv {txts} {direc}/.', shell = True)

        # save_U(U, pr, 'post_sol_proj', pm, direc = direc)

        return fields