# Parameters
Lx = 1.0 # Longitud del dominio
Ly = 1.0 # Longitud del dominio
T = 1500  # Tiempo total de simulación
stat = 499700
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
T_guess = 6.15
t_guess = 780.0
sx_guess = 0.
N_newt = 200
N_gmres = 50
N_hook = 10
tol_newt = 1e-5
tol_gmres = 1e-5
hook = True
vort_X = True