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
ffreq = 20 #fields freq
bfreq = 20 #balance freq
rkord = 2 # order of RK

#Parameters for Newton-Solver

#Get T,t from txt file from directory name
# T_guess = 2.15
# t_guess = 780.8
sx = 0.
N_newt = 200
N_gmres = 50
N_hook = 15
restart = 0 # 0 if not restarting, else, last newton iteration to restart from
tol_newt = 1e-5 #tolerance
tol_gmres = 1e-5
tol_eig = 1e-20
N_eig = 250
hook = True #if true performs hookstep
ls = False #if true performs linesearch

vort_X = False #if true saves vorticity in X vector 
fvort = False #if true saves fourier of vorticity in X vector