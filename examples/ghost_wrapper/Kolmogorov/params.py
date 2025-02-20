# Kolmogorov parameters
L = 1.0 # Domain length (in multiples of 2pi)
T = 5.0 # Evolution time (dummy variable, changed throughout code)
Nx = 256 # Number of grid points in x
Ny = 256 # Number of grid points in y
nu = 1./40. # Kinematic viscosity
dt = 1e-3 # Time step
rkord = 2 # Order of Runge-Kutta method
precision = 'double' # Precision of the code
bstep = 10 # Time step for saving text files
ostep = 10 # Time step for saving output files
sstep = 0 # Time step for saving spectra
ext = 5 # Number of digits in file names
nprocs = 10 # Number of processors
ipath = 'outkol_ghost' # Input path
opath = 'outkol_ghost' # Output path
bpath = 'outkol_ghost' # Balance path
spath = 'spectra' # Spectra path