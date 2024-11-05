import json
import numpy as np
import matplotlib.pyplot as plt
import os

import jax.numpy as jnp
import optax

from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D


# Save hb and dg arays in file
def save_memmap(filename, new_data, iit, total_iterations, dtype=np.float64):
    """
    Saves new data to an existing or new preallocated memory-mapped .npy file.

    Args:
        filename (str): Path to the memory-mapped .npy file.
        new_data (np.ndarray): Data to be saved at the current iteration.
        iit (int): Current iteration (used to index the memory-mapped file).
        total_iterations (int): Total number of iterations to preallocate space for.
        dtype (type): Data type of the saved array (default: np.float64).
    """
    if iit == 0:
        # Create a new memory-mapped file with preallocated space for all iterations
        if os.path.exists(filename):
            os.remove(filename)
        # Shape includes space for total_iterations along the first axis
        shape = (total_iterations,) + new_data.shape  # Preallocate for total iterations
        fp = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
    else:
        # Load the existing memory-mapped file (no need to resize anymore)
        fp = np.load(filename, mmap_mode='r+')

    # Write new data into the current iteration slot
    fp[iit] = new_data
    del fp  # Force the file to flush and close

def update_loss_file(loss_file, iit0):
    '''Kills lines bigger than the starting iteration step in .dat files'''
    # Step 1: Read the existing loss file, if it exists
    filtered_lines = []
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            for line in f:
                iter_num = int(line.split()[0])  # Get the iteration number from each line
                if iter_num < iit0:
                    filtered_lines.append(line)  # Keep lines with iterations less than current iit

    # Step 2: Write back the filtered lines to the file (overwrite it)
    with open(loss_file, 'w') as f:
        f.writelines(filtered_lines)
