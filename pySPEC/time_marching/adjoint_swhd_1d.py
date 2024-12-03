''' 1D Adjoint Shallow Water Equations '''

import numpy as np
import os

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class Adjoint_SWHD_1D(PseudoSpectral):
    ''' 1D Adjoint Shallow Water Equations
        ut_ + u ux_ + (u*u_)x + (h-hb) hx_ = 2(u-um)
        ht_ + u hx_ + g ux_  = 2(h-hm)
    where u,h are physical velocity and height fields,
    u_,h_ the adjoint state fields,
    um,hm the physical field measurements,
    and hb is bottom topography condition.
    '''

    num_fields = 2 # adoint fields u_ , h_
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)
        self.iit = pm.iit
        # self.sample_rate = pm.sample_rate
        self.dt_step = pm.dt_step
        self.dx_step = pm.dx_step
        self.total_steps =  round(self.pm.T/self.pm.dt)
        self.data_path = pm.data_path
        self.field_path = pm.field_path
        self.hb_path = pm.hb_path
        # self.hb = np.load(f'{self.hb_path}/hb_{self.iit}.npy') # hb field at current GD iteration
        self.hb = np.load(f'{self.hb_path}/hb_memmap.npy', mmap_mode='r')[self.iit-1]  # Access the data at the current iteration and Load hb at current GD iteration
        try:
            self.uus =  np.load(f'{self.field_path}/uu_memmap.npy', mmap_mode='r') # all uu fields in time
        except:
            self.uus = None
        try:
            self.hhs =  np.load(f'{self.field_path}/hh_memmap.npy', mmap_mode='r') # all hh fields in time
        except:
            self.hhs = None
        self.uums, self.uusparse_time, self.uusparse_space = self.sample_memmap(data_path = self.data_path, filename = 'uu_memmap.npy',dt_step = self.dt_step, dx_step = self.dx_step)
        self.hhms, self.hhsparse_time, self.hhsparse_space = self.sample_memmap(data_path = self.data_path, filename = 'hh_memmap.npy',dt_step = self.dt_step, dx_step = self.dx_step)
        # make sparse forcing terms
        uus_sparse, sparse_time, sparse_space = self.sample_memmap(data_path = self.field_path, filename = 'uu_memmap.npy',dt_step = self.dt_step, dx_step = self.dx_step)
        hhs_sparse, sparse_time, sparse_space = self.sample_memmap(data_path = self.field_path, filename = 'hh_memmap.npy',dt_step = self.dt_step, dx_step = self.dx_step)
        self.uuforcings = uus_sparse -self.uums
        self.hhforcings = hhs_sparse -self.hhms

    def rkstep(self, fields, prev, oo):
        # Unpack
        fu_  = fields[0]
        fu_p = prev[0]

        fh_  = fields[1]
        fh_p = prev[1]

        # Access the step from the evolve method via the class attribute
        step = self.current_step

        # get physical fields and measurements from T to t0, back stepping in time
        Nt = round(self.pm.T/self.pm.dt)
        back_step = Nt-1 - step
        # uu = np.load(f'{self.field_path}/uu_{back_step:04}.npy') # u field at current time step
        uu = self.uus[back_step]
        # hh = np.load(f'{self.field_path}/hh_{back_step:04}.npy') # h field at current time step
        hh = self.hhs[back_step]

        # get hb
        hb = self.hb

        # measurements
        uum = self.uums[back_step]
        hhm = self.hhms[back_step]

        # sparse forcing terms
        uuforcing = self.uuforcings[back_step]
        hhforcing = self.hhforcings[back_step]

        fuuforcing = self.grid.forward(uuforcing)
        fhhforcing = self.grid.forward(hhforcing)

        fu  = self.grid.forward(uu)
        fux = self.grid.deriv(fu, self.grid.kx)
        ux = self.grid.inverse(fux)

        # calculate terms
        uu_ = self.grid.inverse(fu_)
        # hh_ = self.grid.inverse(fh_)

        fux_ = self.grid.deriv(fu_, self.grid.kx)
        ux_ = self.grid.inverse(fux_)
        fhx_ = self.grid.deriv(fh_, self.grid.kx)
        hx_ = self.grid.inverse(fhx_)

        fu_ux_ = self.grid.forward(uu*ux_)
        fu_u_x = self.grid.deriv(self.grid.forward(uu*uu_), self.grid.kx)
        fu_hx_ = self.grid.forward(uu*hx_)

        fh_hb_hx_ = self.grid.forward((hh-hb)*hx_)

        # backwards integration in time, with sampled forcing terms
        fu_ = fu_p - (self.grid.dt/oo) * (2*fuuforcing - fu_ux_ - fu_u_x - fh_hb_hx_)
        fh_ = fh_p - (self.grid.dt/oo) * (2*fhhforcing - self.pm.g*fux_ - fu_hx_)

        # save zero modes for debugging
        # np.save(f'{self.pm.out_path}/adjoint_zero_modes_{step:04}', np.array([fu_[self.grid.zero_mode], fh_[self.grid.zero_mode]]))


        # de-aliasing
        # fu_[self.grid.zero_mode] = 0.0 # It would seem u_ should have mode zero
        fu_[self.grid.dealias_modes] = 0.0
        fh_[self.grid.zero_mode] = 0.0 # It would seem h_ should not have mode zero
        fh_[self.grid.dealias_modes] = 0.0

        # GD step
        # update new hx_
        fhx_ = self.grid.deriv(fh_, self.grid.kx)
        hx_ = self.grid.inverse(fhx_)
        # multiply with field
        non_dealiased_dg_ = hx_* uu
        fdg_ = self.grid.forward(non_dealiased_dg_)
        # de-aliasing GD step
        fdg_[self.grid.dealias_modes] = 0.0
        dg_ = self.grid.inverse(fdg_)
        # np.save(f'{self.pm.out_path}/hx_uu_{step:04}', dg_)
        self.save_memmap(f'{self.pm.out_path}/hx_uu_memmap.npy', dg_, step, self.total_steps, dtype=np.float64)

        # alternative GD step
        # update new h_
        h_ = self.grid.inverse(fh_)
        # multiply with field
        non_dealiased_dg = h_* ux
        fdg = self.grid.forward(non_dealiased_dg)
        # de-aliasing GD step
        fdg[self.grid.dealias_modes] = 0.0
        dg = self.grid.inverse(fdg)
        # np.save(f'{self.pm.out_path}/h_ux_{step:04}', dg)
        self.save_memmap(f'{self.pm.out_path}/h_ux_memmap.npy', dg, step, self.total_steps, dtype=np.float64)

        return [fu_,fh_]

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

    def sample_memmap(self, data_path, filename, dt_step = 1, dx_step = 1):
        '''returns sparse space-time measurements given dt and dx'''

        # Load the memory-mapped field
        field = np.load(f'{data_path}/{filename}', mmap_mode='r')[:self.total_steps, :]  # read-only measurements until time T

        # Create a copy to avoid modifying the original
        modified_data_ = np.zeros_like(field)

        # Identify indices to keep based on the time interval
        time_indices = np.arange(0, field.shape[0], dt_step)
        # Apply the time-interval-based filter
        modified_data_[time_indices] = field[time_indices]

        # this is the pinn-like random sampling of buoys
        # Randomly choose spatial indices to replace with zero (non-buoyed spaces)
        # random_indices = np.random.choice(field.shape[-1], size=int(field.shape[-1] * (1 - sample_rate)), replace=False)
        # Replace those indices with zero in the copy
        # modified_data[:,  zero_indices] = 0

        # Create another copy to avoid modifying the original
        modified_data = np.zeros_like(modified_data_)
        # Identify indices to keep based on the space interval
        space_indices = np.arange(0, field.shape[-1], dx_step)
        modified_data[:,space_indices] = modified_data_[:,space_indices]
        return modified_data, time_indices, space_indices



    def outs(self, fields, step):
        uu_ = self.grid.inverse(fields[0])
        # np.save(f'{self.pm.out_path}/adjuu_{step:04}', uu_)
        self.save_memmap(f'{self.pm.out_path}/adjuu_memmap.npy', uu_, step, self.total_steps, dtype=np.float64)
        hh_ = self.grid.inverse(fields[1])
        # np.save(f'{self.pm.out_path}/adjhh_{step:04}', hh_)
        self.save_memmap(f'{self.pm.out_path}/adjhh_memmap.npy', hh_, step, self.total_steps, dtype=np.float64)


    def balance(self, fields, step):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)
