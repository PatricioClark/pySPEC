# BOUSS parameters
Lx = 1.0 # Domain length in x (in multiples of 2pi)
Lz = 3.14159 # Domain length in z
Nx = 256 # Number of grid points in x
Ny = 1 # Number of grid points in y
Nz = 103 # Number of grid points in z
dt = 5e-4 # Time step
ra = 1e6 # Rayleigh number
bstep = 100 # Time step for saving text files
ostep = 200 # Time step for saving output files
stat = 0 # Time step of last file if restarting
ext = 5 # Number of digits in file names
nprocs = 28 # Number of processors dedicated to simulation

# Newton-Solver parameters 
T = 12.0 # Initial guess of UPO period
sx = 0. # Initial guess of UPO shift in x. If None then RPOs are not searched for  
restart = 0 # Last completed Newton iteration if restarting
input = "input/" # Path to input files
start_idx = 2000 # Index of input files 

N_newt = 200 # Maximum number of Newton iterations
N_gmres = 300 # Maximum number of GMRES iterations
tol_newt = 1e-10 # Tolerance for Newton iterations
tol_gmres = 1e-3 # Tolerance for GMRES iterations

glob_method = 1 # Newton global method: 0 for no glob method, 1 for hookstep
N_hook = 25 # Maximum number of hookstep iterations
c = 0.5 # Parameter for regularization condition
reduc_reg = 0.5 # Reduction factor of trust region
mu0 = 1e-6 # Initial regularization parameter
mu_inc = 1.5 # Increase factor for regularization

sp1 = False # Performs solenoidal projection in GMRes perturbation
sp2 = False # Performs solenoidal projection in Newton perturbation
sp_dU = False # Performs solenoidal projection over dU instead of U+dU
cmplx = False # Use complex velocity fields for Newton solver
tol_nudge = 1e-3 # Tolerance for selecting different initial condition from orbit if solution is not converging
frac_nudge = 0. # Fraction of period to nudge initial condition
