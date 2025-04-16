''' 2D Rayleigh-Benard solver '''

import numpy as np
import subprocess
import os

from .solver import Solver
from .. import pseudo as ps

class GHOST(Solver):
    '''
    GHOST 2D/3D flows solver
    '''

    def __init__(self, pm, solver = 'HD', dimension = 2):
        self.solver = solver
        self.dimension = dimension

        if self.dimension == 2:
            self.ftypes = ['uu', 'vv']
            self.grid = ps.Grid2D(pm)
            self.num_fields = 2
            self.dim_fields = 2
        elif self.dimension == 3:
            self.ftypes = ['vx', 'vy', 'vz']
            self.grid = ps.Grid3D(pm)
            self.num_fields = 3
            self.dim_fields = 3
        else:
            raise ValueError('Invalid dimension')
        super().__init__(pm)

    def vel_to_ps(self, fields):
        '''Converts velocity fields to stream function'''
        # Compute vorticity field
        fu, fv = [self.grid.forward(ff) for ff in fields]
        uy = self.grid.deriv(fu, self.grid.ky)
        vx = self.grid.deriv(fv, self.grid.kx)
        foz = vx - uy
        return self.grid.inverse(np.divide(foz, self.grid.k2, out = np.zeros_like(foz), where = self.grid.k2!=0.))

    def ps_to_vel(self, ps):
        '''Converts stream function to velocity fields'''
        fps = self.grid.forward(ps)
        fu = self.grid.deriv(fps, self.grid.ky)
        fv = -self.grid.deriv(fps, self.grid.kx)
        fields = self.grid.inverse(fu), self.grid.inverse(fv)
        return fields

    def evolve(self, fields, T, ipath=None, opath = '.', bpath='.', spath='.', ostep=0, bstep=0, sstep=0, vort = False):
        '''Evolves fields in T time. Calls Fortran'''
        if ipath is None:
            ipath = opath
            
        self.write_fields(fields, path = ipath)
        self.ch_params(T, ipath, opath, bstep, ostep, sstep, vort) #save fields every ostep, bal every bstep, spectrum every sstep

        #run GHOST
        subprocess.run(f'mpirun -n {self.pm.nprocs} ./{self.solver}', shell = True)

        #save balance prints
        if bstep:
            txts = 'balance.txt'
            subprocess.run(f'mv {txts} {bpath}/.', shell = True)
        #save spectra prints
        if sstep:
            txts = 'kspectrum* ktransfer*'
            subprocess.run(f'mv {txts} {spath}/.', shell = True)

        #load evolved fields
        if not ostep:
            fields = self.load_fields()
        else:
            fields = self.load_fields(path=opath, idx = int(T/self.pm.dt //ostep) + 1) # +1 since we start from 1
        return fields

    def save_binary_file(self, path, data):
        '''writes fortran file'''
        dtype = np.float64 if self.pm.precision == 'double' else np.float32
        data = data.astype(dtype).reshape(data.size,order='F')
        data.tofile(path)

    def write_fields(self, fields, path, stat=1):
        ''' Writes fields to binary file. Saves temporal fields with idx=1'''
        if self.dimension == 2:
            field = self.vel_to_ps(fields)
            self.save_binary_file(os.path.join(path,f'ps.{stat:0{self.pm.ext}}.out'), field)
        else:
            for field, ftype in zip(fields, self.ftypes):
                self.save_binary_file(os.path.join(path,f'{ftype}.{stat:0{self.pm.ext}}.out'), field)

    def load_fields(self, path = '.', idx = 2): 
        '''Loads binary fields. idx = 2 for default read '''
        dtype = np.float64 if self.pm.precision == 'double' else np.float32
        if self.dimension == 2:
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

    def ch_params(self, T, ipath, opath, bstep = 0, ostep=0, sstep = 0, vort = False, stat = 1):
        '''Changes parameter to update params throughout algorithm '''
        if self.dimension == 2:
            suffix = 'txt'
        elif self.dimension == 3:
            suffix = 'inp'
        with open(f'parameter.{suffix}', 'r') as file:
            lines = file.readlines()

        if ostep == 0:
            ostep = int(T/self.pm.dt)

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
                lines[i] = f'step = {int(T/self.pm.dt)+1}      ! total number of steps\n'
            if line.startswith('cstep'): #modify cstep (bstep in current code)
                lines[i] = f'cstep = {bstep} !steps between writing global quantities\n'
            if line.startswith('sstep'): #modify cstep (bstep in current code)
                lines[i] = f'sstep = {sstep} !number of steps between spectrum output\n'
            if line.startswith('tstep'): #modify tstep (ostep in current code)
                lines[i] = f'tstep = {ostep} !steps between saving fields\n'
            if line.startswith('outs'):
                if vort: # to save additional vorticity fields
                    lines[i] = 'outs = 1   ! controls the amount of output\n'
                else: 
                    lines[i] = 'outs = 0   ! controls the amount of output\n'

        #write
        with open(f'parameter.{suffix}', 'w') as file:
            file.writelines(lines)


    def ch_params2(self, T, ipath, opath, bstep = 0, ostep=0, sstep = 0, vort = False, stat = 1):
        'TODO: implementation of GHOST_dt'
        '''Applies when running with GHOST_dt, which allows evolving with last variable dt and saves final fields '''
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
                lines[i] = f'step = {int(T//self.pm.dt)+1}      ! total number of steps\n'
            if line.startswith('cstep'): #modify cstep (bstep in current code)
                lines[i] = f'cstep = {bstep} !steps between writing global quantities\n'
            if line.startswith('sstep'): #modify cstep (bstep in current code)
                lines[i] = f'sstep = {sstep} !number of steps between spectrum output\n'
            if line.startswith('tstep'): #modify tstep (ostep in current code)
                lines[i] = f'tstep = {ostep} !steps between saving fields\n'
            if line.startswith('nu'): #modifies ra (does not change throughout algorithm)
                lines[i] = f'nu = {self.pm.nu}       ! kinematic viscosity\n'
            if line.startswith('outs'):
                if vort: # to save additional vorticity fields
                    lines[i] = 'outs = 1   ! controls the amount of output\n'
                else: 
                    lines[i] = 'outs = 0   ! controls the amount of output\n'

        #write
        with open('parameter.txt', 'w') as file:
            file.writelines(lines)
