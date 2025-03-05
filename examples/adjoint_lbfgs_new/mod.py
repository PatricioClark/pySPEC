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

def update_npy_file(npy_path, iit0):
    file = np.load(f'{npy_path}')
    storage = np.full(file.shape, np.nan, dtype=np.float64)
    storage[:iit0] = file[:iit0]
    return storage

def reset(fpm, bpm, fsolver, bsolver):
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
        print('remove backward content')
        for filename in os.listdir(bpm.out_path):
            file_path = os.path.join(bpm.out_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        # Initial hb ansatz
        hb = np.zeros_like(fsolver.true_hb)
        dg = np.zeros_like(fsolver.true_hb)
    else:
        fpm.iit = fpm.iit0
        bpm.iit = bpm.iit0
        # handle .dat files if restarting from a previous run

        bsolver.u_loss = update_npy_file(fpm.hb_path+'/u_loss.npy', bpm.iit0)
        bsolver.h_loss = update_npy_file(fpm.hb_path+'/h_loss.npy', bpm.iit0)
        bsolver.val = update_npy_file(fpm.hb_path+'/validation.npy', bpm.iit0)
        bsolver.hbs = update_npy_file(fpm.hb_path+'/hbs.npy', bpm.iit0)
        bsolver.dgs = update_npy_file(fpm.hb_path+'/dgs.npy', bpm.iit0)
        hb = bsolver.hbs[bpm.iit0-1]
        dg = bsolver.dgs[bpm.iit0-1]
    return hb, dg

def early_stopping(w, loss, iit, patience=3):
    """
    Simple early stopping function.

    Args:
        w (int): Current patience counter.
        loss (list): List of loss values.
        patience (int): Number of steps to wait before stopping.

    Returns:
        (int, bool): Updated patience counter and stopping signal.
    """

    # Check if the loss improved
    print('loss compare: ' , loss[iit-1] , loss[iit-2])
    if loss[iit-1] <= loss[iit-2]:
        w = 0  # Reset patience
    else:
        w += 1

    # Stop if patience runs out
    if w >= patience:
        print(f"Stopping early after {w} iterations without improvement.")
        return w, True  # Signal to stop

    return w, False  # Keep going


def plot_fields(fpm,
                hb,
                true_hb,
                out_u,
                true_u,
                out_h,
                true_h,
                noise_u=None,
                noise_h=None):
    f,axs = plt.subplots(ncols=3, figsize = (15,5))
    axs[0].plot(np.linspace(0,2*np.pi, len(hb)), hb , color = 'blue', label = '$\hat{h_b}$')
    axs[0].plot(np.linspace(0,2*np.pi, len(true_hb)), true_hb , alpha = 0.6, color = 'green', label = '$h_b$')
    axs[0].legend(fontsize=14)
    axs[0].set_xlabel('x', fontsize=12)
    axs[1].plot(np.linspace(0,2*np.pi, len(out_u)), out_u, color = 'blue', linestyle= '--', label = '$\hat{u}$')
    axs[1].plot(np.linspace(0,2*np.pi, len(true_u)), true_u, alpha = 1, color = 'green', label = '$u$')
    if fpm.noise:
        axs[1].plot(np.linspace(0,2*np.pi, len(noise_u)), noise_u, alpha = 0.5, color = 'red', label = '$u+\epsilon$')

    axs[1].legend(fontsize=14)
    axs[1].set_xlabel('x', fontsize=12)
    axs[2].plot(np.linspace(0,2*np.pi, len(out_h)), out_h, color = 'blue', linestyle= '--', label = '$\hat{h}$')
    axs[2].plot(np.linspace(0,2*np.pi, len(true_h)), true_h, alpha = 1, color = 'green', label = '$h$')
    if fpm.noise:
        axs[2].plot(np.linspace(0,2*np.pi, len(noise_h)), noise_h, alpha = 0.5, color = 'red', label = '$h+\epsilon$')

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


def plot_loss(fpm, u_loss, h_loss, val):
    plt.figure()
    plt.semilogy(u_loss, label = '$(u-\hat{u})^2$')
    plt.semilogy(h_loss, label = '$(h- \hat{h})^2$')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('Square Error', fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'{fpm.hb_path}/loss.png')

    plt.figure()
    plt.semilogy(val, label = '$(h_b-\hat{h_b})^2$')
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

def plot_hbs(fpm, true_hb, hbs):
    plt.figure()
    plt.plot(np.linspace(0,2*np.pi, len(hbs[0])), hbs[0] , color = 'blue', linestyle = '--', alpha = 0.7, label = '$\hat{h_b}$')
    for iteration in np.arange(0,fpm.iitN, fpm.ckpt):
        plt.plot(np.linspace(0,2*np.pi, len(hbs[iteration])), hbs[iteration] , color = 'blue', linestyle = '--', alpha = 0.7)

    plt.plot(np.linspace(0,2*np.pi, len(true_hb)), true_hb , alpha = 0.6, color = 'green', label = '$h_b$')
    plt.legend(fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.savefig(f'{fpm.hb_path}/hbs.png')
