# Parameters
Lx = 1.0 # Longitud del dominio
Ly = 1.0 # Longitud del dominio
T = 1500  # Tiempo total de simulación
stat = 499700 #idx para reinciar simulación
Nx = 256  # Número de puntos de la cuadrícula
Ny = 256  # Número de puntos de la cuadrícula
dt = 1e-3
Nt = int(T/dt)  # Número de pasos de tiempo
nu = 1.0/40.0
print_freq = 10
save_freq = 50
fields_freq = 50 
rkord = 2 # order of RK

#Parameters for Newton-Solver

#Get T,t from txt file from directory name
import numpy as np
import os
wd = os.getcwd()
orb_n = wd.split('/')[-1][3:5]
orbs = np.loadtxt('../readme.txt', delimiter=',', skiprows=1, comments='-', usecols = (0,1,2))
orb = orbs[int(orb_n)]

T_guess = orb[1]
t_guess = orb[2]
# T_guess = 2.15
# t_guess = 780.8
sx_guess = 0.
N_newt = 200
N_gmres = 50
N_hook = 10
restart = 0 # 0 if not restarting, else, last newton iteration to restart from
tol_newt = 1e-5 #tolerance
tol_gmres = 1e-5
hook = True #if true performs hookstep

vort_X = False #if true saves vorticity in X vector 
fvort = False #if true saves fourier of vorticity in X vector