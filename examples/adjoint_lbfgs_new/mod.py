import json
import numpy as np
import matplotlib.pyplot as plt
import os

import jax.numpy as jnp
import optax

from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

import seaborn as sns
sns.set_style("white")
# palette = sns.color_palette("mako", as_cmap=True)
sns.set_palette(palette='Dark2')

def check_dir(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


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



def plot_fields(fpm,hb,
                true_hb,
                out_u,
                out_h):
    f,axs = plt.subplots(ncols=3, figsize = (15,5))
    axs[0].plot(np.linspace(0,2*np.pi, len(hb)), hb , color = 'blue', label = '$\hat{h_b}$')
    axs[0].plot(np.linspace(0,2*np.pi, len(true_hb)), true_hb , alpha = 0.6, color = 'green', label = '$h_b$')
    axs[0].legend(fontsize=14)
    axs[0].set_xlabel('x', fontsize=12)
    axs[1].plot(np.linspace(0,2*np.pi, len(out_u)), out_u , label = '$\hat{u}$')
    axs[1].legend(fontsize=14)
    axs[1].set_xlabel('x', fontsize=12)
    axs[2].plot(np.linspace(0,2*np.pi, len(out_h)), out_h , label = '$\hat{h}$')
    axs[2].legend(fontsize=14)
    axs[2].set_xlabel('x', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/fields.png')

    plt.figure()
    plt.plot(np.sqrt((hb-true_hb)**2) , label = '$\sqrt{(\hat{h_b}-h_b)^2}$')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('RSE', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/hb_error.png')

    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0,2*np.pi, len(hb)) , hb ,label = '$\hat{h_b}$')
    plt.plot(np.linspace(0,2*np.pi, len(hb)) , true_hb ,label = '$h_b$')
    plt.xlabel('x', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/hb.png')

def plot_fourier(fpm, grid, hb,
                true_hb):
    Nx = fpm.Nx
    dx = grid.dx
    # Compute Fourier Transform of hb
    adj_hb_fft = np.fft.fft(hb)
    true_hb_fft = np.fft.fft(true_hb)

    hb_freq = np.fft.fftfreq(Nx, d=dx)

    # Take only the positive frequencies for plotting
    adj_hb_amplitude = np.abs(adj_hb_fft[:Nx // 2])
    true_hb_amplitude = np.abs(true_hb_fft[:Nx // 2])
    hb_freq_positive = hb_freq[:Nx // 2]

    # Plot the Fourier Amplitude Spectrum
    plt.figure(figsize=(10, 3))
    plt.loglog(hb_freq_positive,adj_hb_amplitude , color='green' , label = 'Spectrum of Adjoint $\hat{h_b}$')
    plt.loglog(hb_freq_positive,  true_hb_amplitude, color='black' ,alpha = 1 , label = 'Spectrum of $h_b$')
    plt.legend(fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid()
    plt.savefig(f'{fpm.hb_path}/fft_hb')


def plot_loss(fpm, loss):
    plt.figure()
    plt.semilogy(loss[0], loss[1], label = '$(u-\hat{u})^2$')
    plt.semilogy(loss[0], loss[2], label = '$(h- \hat{h})^2$')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('Square Error', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/loss.png')

    val = np.loadtxt(f'{fpm.hb_path}/hb_val.dat', unpack=True)
    plt.figure()
    plt.semilogy(val[0], val[1], label = '$(h_b-\hat{h_b})^2$')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('Square Error', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/hb_val.png')

def plot_dg(fpm,dg,DG):
    f,ax= plt.subplots(ncols=2, figsize = (15,5))
    ax[0].plot(dg , label = '$\int_{0}^{T} \partial_x h^{\dag} u \,dt$')
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].legend(fontsize=14)
    ax[1].plot(DG , label = '$lbfgs(\int_{0}^{T} \partial_x h^{\dag} u \,dt)$')
    ax[1].set_xlabel('x', fontsize=12)
    ax[1].legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/dg.png')
