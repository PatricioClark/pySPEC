import json
import numpy as np
import matplotlib.pyplot as plt
import os

import jax.numpy as jnp
import optax

from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

def check_dir(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

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

def reset(fpm, bpm):
    '''removes files if iit0 is set to 0. If not, restarts last run from last iit0'''
    if fpm.iit0 == 0:
        print('remove hbs content')
        for filename in os.listdir(fpm.hb_path):
            file_path = os.path.join(fpm.hb_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        print('remove forward content')
        for filename in os.listdir(fpm.out_path):
            file_path = os.path.join(fpm.out_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        print('remove forward content')
        for filename in os.listdir(bpm.out_path):
            file_path = os.path.join(bpm.out_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
    else:
        fpm.iit = fpm.iit0
        bpm.iit = bpm.iit0
        # handle .dat files if restarting from a previous run
        update_loss_file(f'{fpm.hb_path}/loss.dat', bpm.iit0)
        update_loss_file(f'{fpm.hb_path}/hb_val.dat', bpm.iit0)

def get_hb_dg(fpm, bpm, true_hb, total_iterations):
    '''loads last memmap for hb and dg, or creates a new one if they don't exist'''

    # initial hb
    try:
        hb = np.load(f'{bpm.hb_path}/hb_memmap.npy', mmap_mode='r')[bpm.iit0]  # Access the data at the current iteration and Load hb at current GD iteration
        print(f'succesfully grabbed last hb from iit = {fpm.iit0}')
    except:
        print('make initial flat hb for GD')
        hb = np.zeros_like(true_hb)
        save_memmap(f'{fpm.hb_path}/hb_memmap.npy', hb, bpm.iit0,  total_iterations)
        hb = np.load(f'{bpm.hb_path}/hb_memmap.npy', mmap_mode='r')[bpm.iit0]  # Access the data at the current iteration and Load hb at current GD iteration

    # initial dg
    try:
        dg = np.load(f'{bpm.hb_path}/dg_memmap.npy', mmap_mode='r')[bpm.iit0]  # Access the data at the current iteration and Load dg at current GD iteration
        print(f'succesfully grabbed last dg from iit = {fpm.iit0}')
    except:
        print('make initial flat dg for GD')
        save_memmap(f'{fpm.hb_path}/dg_memmap.npy', np.zeros_like(true_hb), bpm.iit0,  total_iterations)
        dg = np.load(f'{bpm.hb_path}/dg_memmap.npy', mmap_mode='r')[bpm.iit0]  # Access the data at the current iteration and Load hb at current GD iteration

    return hb,dg

def plot_fields(fpm,hb,
                true_hb,
                out_u,
                out_h):
    f,axs = plt.subplots(ncols=3, figsize = (15,5))
    axs[0].plot(hb , color = 'blue', label = 'pred hb')
    axs[0].plot(true_hb , alpha = 0.6, color = 'green', label = 'true hb')
    axs[0].legend()
    axs[1].plot(out_u , label = 'u')
    axs[1].legend()
    axs[2].plot(out_h , label = 'h')
    axs[2].legend()
    plt.savefig(f'{fpm.hb_path}/fields.png')

    plt.figure()
    plt.plot(np.sqrt((hb-true_hb)**2) , label = 'hb error')
    plt.savefig(f'{fpm.hb_path}/hb_error.png')

def plot_loss(fpm, loss):
    plt.figure()
    plt.plot(loss[0], loss[1], label = '$(u-\hat{u})^2$')
    plt.plot(loss[0], loss[2], label = '$(h- \hat{h})^2$')
    plt.legend()
    plt.savefig(f'{fpm.hb_path}/loss.png')

    val = np.loadtxt(f'{fpm.hb_path}/hb_val.dat', unpack=True)
    plt.figure()
    plt.plot(val[0], val[1], label = '$(hb-\hat{hb})^2$')
    plt.legend()
    plt.savefig(f'{fpm.hb_path}/hb_val.png')
