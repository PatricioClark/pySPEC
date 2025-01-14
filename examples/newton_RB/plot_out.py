import numpy as np 
import matplotlib.pyplot as plt
import json

import pySPEC as ps
from pySPEC.time_marching import KolmogorovFlow
from pySPEC.newton import UPO
import params as pm

pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
grid   = ps.Grid2D(pm)
solver = KolmogorovFlow(pm)


def plot_fields(path, step):
    '''Plot output'''
    uu = np.load(f'{path}uu_{step:0{pm.ext}}.npy')
    vv = np.load(f'{path}vv_{step:0{pm.ext}}.npy')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(uu, cmap='viridis')
    ax[0].set_title('uu')
    # ax[0].axis('off')
    ax[1].imshow(vv, cmap='viridis')
    ax[1].set_title('vv')
    # ax[1].axis('off')
    plt.savefig(f'{path}fields_{step}.png', dpi = 300)
    plt.show()

def plot_oz(path, step, ext = pm.ext, convert = True):	
    '''Plot output'''
    if convert:
        uu = np.load(f'{path}uu_{step:0{ext}}.npy')
        vv = np.load(f'{path}vv_{step:0{ext}}.npy')

        oz = solver.oz([uu, vv])
        # oz = mod.vort(uu, vv, pm, grid)
    else:
        oz = np.load(f'{path}oz_{step:0{ext}}.npy')

    plt.figure()
    plt.imshow(oz.T, cmap='viridis')
    plt.title('oz')
    plt.savefig(f'{path}oz_{step}.png', dpi = 300)
    plt.show()


def plot_forcing():
    '''Plot forcing'''
    # self.fx = np.sin(2*np.pi*kf*self.grid.yy/pm.Ly)
    # self.fx = self.grid.forward(self.fx)
    # self.fy = np.zeros_like(self.fx, dtype=complex)
    # self.fx, self.fy = self.grid.inc_proj([self.fx, self.fy])

    fx = grid.inverse(solver.fx)
    fy = grid.inverse(solver.fy)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(fx.T, cmap='viridis')
    ax[0].set_title('fx')
    ax[1].imshow(fy.T, cmap='viridis')
    ax[1].set_title('fy')
    plt.savefig(f'forcing.png', dpi = 300)
    plt.show()


def calc_eng(path, step):
    '''Calcualte energy'''
    uu = np.load(f'{path}uu_{step:0{pm.ext}}.npy')
    vv = np.load(f'{path}vv_{step:0{pm.ext}}.npy')
    fu, fv = grid.forward([uu, vv])
    eng = grid.energy([fu, fv])
    print(f'Energy: {eng}')



if __name__ == '__main__':
    path = 'output/iN00/'
    # path = 'input/'
    step = 0
    ext = pm.ext
    # plot_fields(path, step)
    # plot_oz(path, step, ext, True)
    # plot_forcing()

    calc_eng(path, step)    


