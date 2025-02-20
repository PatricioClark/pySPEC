''' 2D Rayleigh-Benard solver '''

import numpy as np
import subprocess
import os

from .solver import Solver
from .. import pseudo as ps

class GHOST(Solver):
    '''
    GHOST 2D flows solver
    '''

    num_fields = 2
    dim_fields = 2

    def __init__(self, pm, solver = 'HD'):
        super().__init__(pm)
        self.grid = ps.Grid2D(pm)
        self.solver = solver

        if self.solver == 'HD':
            self.ftypes = ['uu', 'vv']
        else:
            raise ValueError('Invalid solver')

    def vel_to_ps(self, fields):
        '''Converts velocity fields to stream function'''
        # Compute vorticity field
        fu, fv = [self.grid.forward(ff) for ff in fields]
        uy = self.grid.deriv(fu, self.grid.ky)
        vx = self.grid.deriv(fv, self.grid.kx)
        foz = uy - vx
        return self.grid.inverse(np.divide(foz, self.grid.k2, out = np.zeros_like(foz), where = self.grid.k2!=0.))

    def ps_to_vel(self, ps):
        '''Converts stream function to velocity fields'''
        fps = self.grid.forward(ps)
        fu = self.grid.deriv(fps, self.grid.ky)
        fv = -self.grid.deriv(fps, self.grid.kx)
        fields = self.grid.inverse(fu), self.grid.inverse(fv)
        return fields

    def evolve(self, fields, T, ipath='.', opath = '.', bstep=None, ostep=None, sstep=None, bpath='.', spath='.'):
        '''Evolves fields in T time. Calls Fortran'''
        self.write_fields(fields, path = ipath)

        if ostep is None:
            self.ch_params(T, ipath, opath) #change period to evolve
        else:
            self.ch_params(T, ipath, opath, bstep=bstep, ostep=ostep, sstep=sstep) #save fields every ostep, and bal every bstep

        #run GHOST
        subprocess.run(f'mpirun -n {self.pm.nprocs} ./{self.solver}', shell = True)

        #save balance prints
        if bstep is not None:
            txts = 'balance.txt'
            subprocess.run(f'mv {txts} {bpath}/.', shell = True)
        #save spectra prints
        if sstep is not None:
            txts = 'kspectrum* ktransfer*'
            subprocess.run(f'mv {txts} {spath}/.', shell = True)

        #load evolved fields
        if ostep is None:
            fields = self.load_fields()
        else:
            fields = self.load_fields(path=opath, idx = int(T/self.pm.dt //ostep))
        return fields

    def save_binary_file(self, path, data):
        '''writes fortran file'''
        dtype = np.float64 if self.pm.precision == 'double' else np.float32
        data = data.astype(dtype).reshape(data.size,order='F')
        data.tofile(path)

    def write_fields(self, fields, path, stat=1):
        ''' Writes fields to binary file. Saves temporal fields with idx=1'''
        if self.solver == 'HD':
            field = self.vel_to_ps(fields)
            self.save_binary_file(os.path.join(path,f'ps.{stat:0{self.pm.ext}}.out'), field)
        else:
            for field, ftype in zip(fields, self.ftypes):
                self.save_binary_file(os.path.join(path,f'{ftype}.{stat:0{self.pm.ext}}.out'), field)

    def load_fields(self, path = '.', idx = 2): 
        '''Loads binary fields. idx = 2 for default read '''
        dtype = np.float64 if self.pm.precision == 'double' else np.float32
        if self.solver == 'HD':
            ftype = 'ps'
            file = os.path.join(path,f'{ftype}.{idx:0{self.pm.ext}}.out')
            ps = np.fromfile(file,dtype=dtype).reshape(self.grid.shape,order='F')
            fields = self.ps_to_vel(ps)
        else:
            fields = []
            for ftype in self.ftypes:
                file = os.path.join(path,f'{ftype}.{idx:0{self.pm.ext}}.out')
                fields.append(np.fromfile(file,dtype=dtype).reshape(self.grid.shape,order='F'))
        return fields

    def ch_params(self, T, ipath, opath, stat = 1, bstep = 0, ostep=0, sstep = 0):
        '''Changes parameter.txt to update T, and sx '''
        with open('parameter.txt', 'r') as file:
            lines = file.readlines()

        if ostep == 0:
            ostep = int(T//self.pm.dt)

        for i, line in enumerate(lines):
            if line.startswith('idir'): #modifies input directory
                lines[i] = f'idir = "{ipath}" \n'
            if line.startswith('odir'): #modifies output directory
                lines[i] = f'odir = "{opath}" \n'
            if line.startswith('stat'): #modifies starting index
                lines[i] = f'stat = {stat}    ! last binary file if restarting an old run\n'
            if line.startswith('dt'): #modifies dt (does not change throughout algorithm)
                lines[i] = f'dt = {self.pm.dt}   ! time step\n'
            if line.startswith('step'):#modify period:
                lines[i] = f'step = {int(T//self.pm.dt) + 1}      ! total number of steps\n'
            if line.startswith('cstep'): #modify cstep (bstep in current code)
                lines[i] = f'cstep = {bstep} !steps between writing global quantities\n'
            if line.startswith('sstep'): #modify cstep (bstep in current code)
                lines[i] = f'sstep = {sstep} !number of steps between spectrum output\n'
            if line.startswith('tstep'): #modify tstep (ostep in current code)
                lines[i] = f'tstep = {ostep} !steps between saving fields\n'
            if line.startswith('nu'): #modifies ra (does not change throughout algorithm)
                lines[i] = f'nu = {self.pm.nu}       ! kinematic viscosity\n'

        #write
        with open('parameter.txt', 'w') as file:
            file.writelines(lines)
