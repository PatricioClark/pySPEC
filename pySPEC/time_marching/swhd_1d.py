''' 1D Shallow Water Equations '''

import numpy as np
import os

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class SWHD_1D(PseudoSpectral):
    ''' 1D Shallow Water Equations
        ut + u ux + g hx = 0
        ht + ( u(h-hb) )x = 0
    where u,h are velocity and height fields,
    and hb is bottom topography condition.
    '''

    num_fields = 2
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)
        self.iit = pm.iit
        self.total_steps =  round(self.pm.T/self.pm.dt)
        self.hb_path = pm.hb_path
        try:
            self.hb = np.load(f'{self.hb_path}/hb_memmap.npy', mmap_mode='r')[self.iit-1]  # Access the data at the current iteration and Load hb at current GD iteration
        except:
            self.hb = np.load(f'{self.hb_path}/hb_{self.iit}.npy') # hb field at current GD iteration


    def rkstep(self, fields, prev, oo):
        # Unpack
        fu  = fields[0]
        fup = prev[0]

        fh  = fields[1]
        fhp = prev[1]

        hb = self.hb
        # save hb stats to debug
        # save hb stats to debug
        # if self.current_step == 2:
        #         loss = [f'{self.iit}', f'{hb.mean():.6e}'  , f'{hb.std():.6e}' ,  f'{hb.max():.6e}' , f'{hb.min():.6e}']
        #         with open(f'{self.pm.hb_path}/front_hb_stats.dat', 'a') as output:
        #             print(*loss, file=output)

        # Non-linear term
        uu = self.grid.inverse(fu)
        hh = self.grid.inverse(fh)

        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))

        fhx = self.grid.deriv(fh, self.grid.kx) # i k_i fh_i

        fu_ux = self.grid.forward(uu*ux)

        fu_h_hb_x = self.grid.deriv(self.grid.forward(uu*(hh-hb)) , self.grid.kx) # i k_i f(uu*(hh-hb))_i

        fu = fup + (self.grid.dt/oo) * (-fu_ux -  self.pm.g*fhx)
        fh = fhp + (self.grid.dt/oo) * (-fu_h_hb_x)

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0
        fu[self.grid.dealias_modes] = 0.0
        # fh[self.grid.zero_mode] = 0.0 # zero mode for h is not null
        fh[self.grid.dealias_modes] = 0.0

        return [fu, fh]

    # Save hb and dg arays in file
    def save_memmap(self, filename, new_data, step, total_steps, dtype=np.float64):
        """
        Saves new data to an existing or new preallocated memory-mapped .npy file.

        Args:
            filename (str): Path to the memory-mapped .npy file.
            new_data (np.ndarray): Data to be saved at the current iteration.
            iit (int): Current iteration (used to index the memory-mapped file).
            total_iterations (int): Total number of iterations to preallocate space for.
            dtype (type): Data type of the saved array (default: np.float64).
        """
        if step == 0:
            # Create a new memory-mapped file with preallocated space for all iterations
            if os.path.exists(filename):
                os.remove(filename)
            # Shape includes space for total_iterations along the first axis
            shape = (total_steps,) + new_data.shape  # Preallocate for total iterations
            fp = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
        else:
            # Load the existing memory-mapped file (no need to resize anymore)
            fp = np.load(filename, mmap_mode='r+')

        # Write new data into the current iteration slot
        fp[step] = new_data
        del fp  # Force the file to flush and close

    def outs(self, fields, step):
        uu = self.grid.inverse(fields[0])
        # np.save(f'{self.pm.out_path}/uu_{step:04}', uu)
        self.save_memmap(f'{self.pm.out_path}/uu_memmap.npy', uu, step, self.total_steps, dtype=np.float64)
        hh = self.grid.inverse(fields[1])
        # np.save(f'{self.pm.out_path}/hh_{step:04}', hh)
        self.save_memmap(f'{self.pm.out_path}/hh_memmap.npy', hh, step, self.total_steps, dtype=np.float64)

    def balance(self, fields, step):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)
