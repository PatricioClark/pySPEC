import json
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import minimize

from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D

def get_s_y_rho(iit , m, hb_path):
    hb = np.load(f'{hb_path}/hb_memmap.npy', mmap_mode='r')[iit - m:iit+1] # take past hb from k-m to k
    dg = np.load(f'{hb_path}/dg_memmap.npy', mmap_mode='r')[iit - m:iit+1] # take past dg from k-m to k
    s = np.diff(hb , axis = 0) # shaped (m , 1024)
    y = np.diff(dg , axis = 0) # shaped (m , 1024)

    # Calculate the dot products for all rows at once
    dot_products = np.einsum('ij,ij->i', y, s)  # This computes (y_k.T s_k) for each k
    rho = 1 / dot_products  # Take the inverse of the dot products

    return s, y, rho


def get_gamma(iit, y, s):
    up = np.dot(s[-2].T , y[-2]) # the position [-1] is k, position [-2] is k-1
    dn = np.dot(y[-2].T , s[-2]) # the position [-1] is k, position [-2] is k-1
    gamma = up/dn # gamma for k = iit
    return gamma

def get_initial_H(iit, y, s):
    gamma = get_gamma(iit, y, s)
    H = gamma * np.identity(n = y.shape[-1]) # gamma times I with nxn dimensions, n matching column vector length
    return H

def lbfgs_recursion_loop1(iit, m, dg, rho, y, s ):

    q = dg.reshape(-1, 1) # reshape so its a column vector
    rho = rho # a list with the last m: 1/(yk.T*sk) column vectors
    s = s # a list with the last m: hbk+1 - hbk column vectors
    y = y # a list with the last m: dgk+1 - dgk column vectors

    a = []

    for i in range(-2, -m-1, -1): # from k-1 to k-m
        si = s[i].reshape(-1,1) # make it into column vector
        yi = y[i].reshape(-1,1) # make it into column vector

        ai = rho[i] * np.dot( si.T , q)[0][0] # a scalar
        a.append(ai) # append alpha for recursion loop 2
        q = q - ai* yi # updated q vector

    a = np.array(a)
    return q, a

def lbfgs_recursion_loop2(iit, m, rho, y, s, q, H, a):

    q = q
    z = np.dot(H , q)
    a = a # list of alpha scalars from recursive loop 1
    rho = rho # a list with the last m: 1/(yk.T*sk) column vectors
    y = y # a list with the last m: xk+1 - xk column vectors
    s = s # a list with the last m: dgk+1 - dgk column vectors

    for i in range(0 , m- 1): # from k-m to k-1
        print('loop 2 i = ' , i)
        si = s[i].reshape(-1,1) # make it into column vector
        yi = y[i].reshape(-1,1) # make it into column vector

        bi = rho[i] * np.dot( yi.T , z) # a scalar
        z = z - si*(a[i] - bi) # updated z vector
    return z # -z will be the gradient descent direction

def lbfgs(iit , m, hb_path, dg):

    print('getting s, y, rho')
    s, y, rho = get_s_y_rho(iit , m, hb_path)
    print('s, y, rho shapes    :   ', s.shape, y.shape, rho.shape)

    print('\ngetting gamma')
    gamma = get_gamma(iit, y, s)
    print('gamma =    :   ', gamma)

    print('\ngetting H')
    H = get_initial_H(iit, y, s)
    print(' H   shape :   ', H.shape)

    print('\n q, alphas')
    q, a = lbfgs_recursion_loop1(iit, m, dg, rho, y, s )
    print(' q, alphas  shape :   ', q.shape, a.shape)

    print('\n z')
    z = lbfgs_recursion_loop2(iit, m, rho, y, s, q, H, a)
    print(' z  shape :   ', z.shape)
    z = z.reshape(1024) # reshape to apply to hb

    return z # -z will be the gradient descent direction
