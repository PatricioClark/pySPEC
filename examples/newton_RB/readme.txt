If starting new Newton Solver specify input folder and index in params.py, this must contain vx, vz, and th.
Initial guess for period and shift (T, sx) must be specified in params.py.
If restarting a Newton Solver from a specific Newton iteration enter the iteration in the restart parameter.
- newton_rb.py runs the newton solver
- floq_exp.py finds the floquet exponents of a converged orbit (Newton iteration and amount of exponents must be specified inside script)
- BOUSS and BOUSS_PROJ are the executable Boussinesq solvers used for the evolution inside the Newton solver
- plot_RB.py plots relevant quantities such as Newton and GMRes errors, plots orbits, balances, and floquet exponents
- tables/ contains the FC-Gram tables used for Fourier continuation in BOUSS
- src/ contains the source code for compiling BOUSS and BOUSS_PROJ