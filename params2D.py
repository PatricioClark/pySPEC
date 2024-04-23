# Parameters
Lx = 1.0 # Longitud del dominio
Ly = 1.0 # Longitud del dominio
T = 500.0  # Tiempo total de simulación
Nx = 256  # Número de puntos de la cuadrícula
Ny = 256  # Número de puntos de la cuadrícula
dt = 1e-3
Nt = int(T/dt)  # Número de pasos de tiempo
nu = 1.0/40.0
print_freq = 10
save_freq = 1000
rkord = 2 # order of RK
