import os
import numpy as np

from .krylov import GMRES, backsub, arnoldi_eig

class DynSys():
    def __init__(self, pm, solver):
        '''
        Parameters:
        ----------
            pm: parameters dictionary
            solver: solver object
        '''
        self.pm = pm
        self.solver = solver
        self.grid = solver.grid

    def load_ic(self):
        ''' Load initial conditions '''
        if not self.pm.restart:
            # Start Newton Solver from initial guess
            fields = self.solver.load_fields(self.pm.input, self.pm.stat)

            T, sx = self.pm.T, self.pm.sx
            # Create directories
            self.mkdirs()
            self.write_header()
        else:
            # Restart Newton Solver from last iteration
            restart_path = f'output/iN{self.pm.restart:02}/'
            fields = self.solver.load_fields(restart_path, 0)
            T, sx = self.get_restart_values(self.pm.restart)

        U = self.flatten(fields)
        X = self.form_X(U, T, sx)
        return X

    def form_X(self, U, T, sx = None, lda = None):
        X = np.append(U, T)
        if sx is not None:
            X = np.append(X, sx)
        if lda is not None:
            X = np.append(X, lda)
        return X

    def unpack_X(self, X, arclength = False):
        '''X could contain extra params sx and lda if searching for RPOs (pm.sx != 0) or using arclength continuation'''
        if not arclength:
            if self.pm.sx is not None:
                U, T, sx = X[:-2], X[-2], X[-1]
            else:
                U, T, sx = X[:-1], X[-1], 0.
            return U, T, sx

        else:            
            if self.pm.sx is not None:
                U, T, sx, lda = X[:-3], X[-3], X[-2], X[-1]
            else: #sets sx to 0
                U, T, sx, lda = X[:-2], X[-2], 0., X[-1]
            return U, T, sx, lda


    def get_restart_values(self, restart, arclength = False):
        ''' Get values from last Newton iteration of T and sx from solver.txt '''
        fname = 'prints/solver.txt'
        iters,T,sx = np.loadtxt(fname, delimiter = ',', skiprows = 1, unpack = True, usecols = (0,2,3))
        idx_restart = np.argwhere(iters==restart)[0][0] #in case more than 1 restart is needed find row with last iter
        values =  [T[idx_restart], sx[idx_restart]]

        if arclength:
            lda = np.loadtxt(fname, delimiter = ',', skiprows = 1, unpack = True, usecols = 5)
            values.append(lda[idx_restart])        
        return values

    def flatten(self, fields):
        '''Flattens fields'''
        return np.concatenate([f.flatten() for f in fields])

    def unflatten(self, U):
        '''Unflatten fields'''
        ll = len(U)//self.solver.num_fields
        fields = [U[i*ll:(i+1)*ll] for i in range(self.solver.num_fields)]
        fields = [f.reshape(self.grid.shape) for f in fields]
        return fields

    def flatten_dec(func):
        """Decorator that allows to work with flattened fields U instead of fields"""
        def wrapper(self, U, *args, **kwargs):
            fields = self.unflatten(U)
            result = func(self, fields, *args, **kwargs)
            return self.flatten(result)
        return wrapper

    @flatten_dec
    def evolve(self, U, T, save = False, iN = 0):
        '''Evolves fields U in time T'''
        if save:
            dir_iN = f'iN{iN:02}/'
            return self.solver.evolve(U, T, self.pm.bstep, self.pm.ostep, bpath = f'balance/{dir_iN}', opath = f'output/{dir_iN}')
        else:
            return self.solver.evolve(U, T)

    @flatten_dec
    def translate(self, U, sx):
        '''Translates fields by sx in x direction'''
        return self.grid.translate(U, sx)

    @flatten_dec
    def deriv_U(self, U):
        '''Derivatives in x direction of fields'''
        return self.grid.deriv_fields(U)

    @flatten_dec
    def sol_project(self, U):
        '''Solenoidal projection of fields'''
        return self.solver.inc_proj(U)

    def apply_proj(self, U, dU, sp):
        '''Applies projection if sp is True. To dU or U+dU'''
        if not sp:
            return U+dU

        if self.pm.sp_dU:
            dU = self.sol_project(dU)
            return U+dU
        else:
            return self.sol_project(U+dU)

    def form_b(self, U, UT):
        "Form RHS of extended Newton system. UT is evolved and translated flattened field"
        b = U - UT
        if self.pm.sx is not None:
            return np.append(b, [0., 0.])
        else:
            return np.append(b, 0.)

    def norm(self, U):
        return np.linalg.norm(U)

    def update_A(self, X, iN):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, T, sx = self.unpack_X(X)

        # Evolve fields and save output
        self.mkdirs_iN(iN-1) # save output and bal from previous iN
        UT = self.evolve(U, T, save = True, iN = iN-1)

        # Translate UT by sx and calculate derivatives
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

            dUT_ds = self.deriv_U(UT)
            dU_ds = self.deriv_U(U)
        else:
            dUT_ds = dU_ds = np.zeros_like(U) # No translation if sx is None

        # Calculate derivatives in time
        dUT_dT = self.evolve(UT, self.pm.dt)
        dUT_dT = (dUT_dT - UT)/self.pm.dt

        dU_dt = self.evolve(U, self.pm.dt)
        dU_dt = (dU_dt - U)/self.pm.dt

        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds = self.unpack_X(dX)

            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU and apply solenoidal projection if sp1 = True
            U_pert = self.apply_proj(U, epsilon*dU, self.pm.sp1)

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon

            # Calculate projections of dU needed for extended Newton system
            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real

            # Save norms for diagnostics
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds, ]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT
            if self.pm.sx is not None:
                return np.append(LHS, [t_proj, Tx_proj])
            else:
                return np.append(LHS, t_proj)

        return apply_A, UT

    def run_newton(self, X):
        '''Iterates Newton-GMRes solver until convergence'''
        for iN in range(self.pm.restart+1, self.pm.N_newt):
            # Unpack X
            U, T, sx = self.unpack_X(X)

            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A(X, iN)

            # RHS of Newton extended system
            b = self.form_b(U, UT)
            F = self.norm(b) #||b|| = ||F||: rootsearch function

            # Write to txts
            self.write_prints(iN, F, U, sx, T)

            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, self.pm.N_gmres, self.pm.tol_gmres, iN, self.pm.glob_method)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN)

            # Update solution
            U, T, sx = self.unpack_X(X)

            # Select different initial condition from orbit if solution is not converging
            if (1-F_new/F) < self.pm.tol_nudge:
                with open('prints/nudge.txt', 'a') as file:
                    file.write(f'iN = {iN}. frac_nudge = {self.pm.frac_nudge}\n')
                U = self.evolve(U, T*self.pm.frac_nudge)

            # Termination condition
            if F_new < self.pm.tol_newt:
                self.mkdirs_iN(iN)
                UT = self.evolve(U, T, save = True, iN = iN)
                if self.pm.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                F = self.norm(b) #||b|| = ||F||: rootsearch function
                # Write to txts
                self.write_prints(iN+1, F, U, sx, T)
                break

    def hookstep(self, X, H, beta, Q, iN, arclength = False):
        ''' Performs hookstep on solution given by GMRes untill new |F| is less than previous |F| (or max iter of hookstep is reached) '''
        # Unpack X
        if not arclength:
            U, T, sx = self.unpack_X(X)
        else:
            U, T, sx, lda = self.unpack_X(X, arclength = True)

        #Initial solution from GMRes in basis Q
        y = backsub(H, beta)
        #Initial trust region radius
        Delta = self.norm(y)

        #Define trust_region function
        trust_region = self.trust_region_function(H, beta, iN, y)

        mu = 0.
        #Perform hookstep
        for iH in range(self.pm.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y #Unitary transform back to full dimension

            if not arclength:
                dU, dT, dsx = self.unpack_X(dx)
            else:
                dU, dT, dsx, dlda = self.unpack_X(dx, arclength = True)

            U_new = self.apply_proj(U, dU, self.pm.sp2)

            sx_new = sx+dsx.real
            T_new = T+dT.real
            X_new = self.form_X(U_new, T_new, sx_new)

            if arclength:
                lda_new = lda+dlda.real
                X_new = np.append(X_new, lda_new)
                self.update_lda(lda_new)

            UT = self.evolve(U_new, T_new)
            if self.pm.sx is not None:
                UT = self.translate(UT,sx_new)
            F_new = self.norm(U_new-UT)

            lin_exp = self.norm(beta - self.pm.c * H @ y) #linear expansion of F around x (in basis Q).
            # beta = H@y holds

            self.write_hookstep(iN, iH, F_new, lin_exp)

            if F_new <= lin_exp:
                break
            else:
                Delta *= self.pm.reduc_reg #reduce trust region
        return X_new, F_new, UT


    def trust_region_function(self, H, beta, iN, y0):
        ''' Performs trust region on solution provided by GMRes. Must be instantiated at each Newton iteration '''
        R = H #R part of QR decomposition
        A_ = R.T @ R #A matrix in trust region
        b = R.T @ beta #b vector in trust region

        def trust_region(Delta, mu):
            ''' Delta: trust region radius, mu: penalty parameter  '''
            if mu == 0: #First hoostep iteration. Doens't perform trust region
                y_norm = self.norm(y0)
                Delta0 = y_norm #Initial trust region

                with open(f'prints/hookstep/extra_iN{iN:02}.txt', 'a') as file:
                    file.write(f'{Delta0},{mu},{y_norm},0\n')

                mu = self.pm.mu0 #Initialize first nonzero value of mu
                return y0, mu

            for _ in range(1, 1000): #1000 big enough to ensure that condition is satisfied

                # Ridge regression adjustment
                A = A_ + mu * np.eye(A_.shape[0])
                y = np.linalg.solve(A, b) #updates solution with trust region

                self.write_trust_region(iN, Delta, mu, y, A)

                if self.norm(y) <= Delta:
                    break
                else:
                    # Increase mu until trust region is satisfied
                    mu *= self.pm.mu_inc

            return y, mu
        return trust_region

    def mkdir(self, path):
        os.makedirs(path, exist_ok=True)

    def mkdirs(self):
        ''' Make directories for solver '''
        dirs = ['output', 'balance', 'prints/error_gmres', 'prints/hookstep', 'prints/apply_A']
        try:
            if self.solver.solver == 'BOUSS':
                dirs.append('bin_tmp')
        except:
            pass

        for dir_ in dirs:
            self.mkdir(dir_)

    def mkdirs_iN(self, iN):
        ''' Directories for specific Newton iteration '''
        dirs = [f'output/iN{iN:02}', f'balance/iN{iN:02}']
        for dir_ in dirs:
            self.mkdir(dir_)

    def write_header(self, arclength = False):
        with open('prints/solver.txt', 'a') as file:
            file.write('Iter newt, |F|, T, sx, |U|')
            if arclength:
                file.write(', lambda, N(X)')
            file.write('\n')

    def write_prints(self, iN, F, U, sx, T, arclength = None):
        with open('prints/solver.txt', 'a') as file1,\
        open(f'prints/error_gmres/iN{iN:02}.txt', 'w') as file2,\
        open(f'prints/hookstep/iN{iN:02}.txt', 'w') as file3,\
        open(f'prints/apply_A/iN{iN:02}.txt', 'w') as file4,\
        open(f'prints/hookstep/extra_iN{iN:02}.txt', 'w') as file5:

            file1.write(f'{iN-1:02},{F:.6e},{T},{sx:.8e},{self.norm(U):.6e}')
            if arclength:
                file1.write(f',{arclength[0]},{arclength[1]}') #writes lambda (arclength[0]) and N(X) (arclength[1])
            file1.write('\n')
            file2.write('iG, error\n')
            file3.write('iH, |F|, |F(x)+cAdx|, |F(x)+Adx|\n')

            if arclength:
                lda_str = ', |dUT_dlda|'
            else:
                lda_str = ''
            file4.write(f'|U|, |dU|, |dUT/dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|{lda_str}, t_proj, Tx_proj')
            if arclength:
                file4.write(f', lda_proj')
            file4.write('\n')
            file5.write('Delta, mu, |y|, cond(A)\n')

    def write_apply_A(self, iN, norms, t_proj, Tx_proj, lda_proj = None):
        with open(f'prints/apply_A/iN{iN:02}.txt', 'a') as file:
            file.write(','.join([f'{norm:.4e}' for norm in norms]) + f',{t_proj:.4e},{Tx_proj:.4e}')
            if lda_proj:
                file.write(f',{lda_proj:.4e}')
            file.write('\n')


    def write_trust_region(self, iN, Delta, mu, y, A):
        with open(f'prints/hookstep/extra_iN{iN:02}.txt', 'a') as file:
            file.write(f'{Delta:.4e},{mu:.4e},{self.norm(y):.4e},{np.linalg.cond(A):.3e}\n')

    def write_hookstep(self, iN, iH, F_new, lin_exp):
        with open(f'prints/hookstep/iN{iN:02}.txt', 'a') as file:
            file.write(f'{iH:02},{F_new:.4e},{lin_exp:.4e}\n')

    def floq_exp(self, X, n, tol, b = 'U'):
        ''' Calculates Floquet exponents of periodic orbit '''
        ''' X: (U,T,sx) of converged periodic orbit, n: number of exponents, tol: tolerance of Arnoldi '''

        from warnings import warn
        warn('This function is deprecated. Use lyap_exp instead.', DeprecationWarning)

        # Unpack X
        U, T, sx = self.unpack_X(X)

        UT = self.evolve(U, T)
        # Translate UT by sx
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

        def apply_J(dU):
            ''' Applies J (jacobian of poincare map) matrix to vector dU  '''
            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU
            U_pert = U + epsilon*dU

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon
            return dUT_dU

        if b == 'U':
            b = U
        elif b == 'random':
            b = np.random.randn(len(U))
        else:
            raise ValueError('b must be U or random')

        eigval_H, eigvec_H, Q = arnoldi_eig(apply_J, b, n, tol)

        return eigval_H, eigvec_H, Q

    def lyap_exp(self, fields, T, n, tol, ep0=1e-7, sx=None, b='U'):
        ''' Calculates Lyapunov exponents of periodic orbit 
        
        Paramters
        ---------
        fields: list of fields.
        T: time to evolve in each arnoldi iteration
        n: number of exponents,
        tol: tolerance of Arnoldi
        ep0: perturbation factor, optional, default=1e-7
        sx: translation in x direction, optional, default=None
        b: initial guess for Arnoldi, optional, default='U'

        Returns
        -------
        eigval_H: Lyapunov exponents
        eigvec_H: Lyapunov vectors
        Q: Arnoldi basis
        '''

        U = self.flatten(fields)
        UT = self.evolve(U, T)

        # Translate UT by sx
        if sx is not None:
            UT = self.translate(UT, sx)

        def apply_J(dU):
            ''' Applies J (jacobian of poincare map) matrix to vector dU  '''
            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = ep0*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU
            U_pert = U + epsilon*dU

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)

            if sx is not None:
                dUT_dU = self.translate(dUT_dU, sx)

            dUT_dU = (dUT_dU - UT)/epsilon
            return dUT_dU

        if isinstance(b, str):
            if b == 'U':
                b = U
            elif b == 'random':
                b = np.random.randn(len(U))
        elif isinstance(b, np.ndarray):
            pass
        else:
            raise ValueError('b must be one of the given options')
        b = np.random.randn(len(U))
        eigval_H, eigvec_H, Q = arnoldi_eig(apply_J, b, n, tol)

        return eigval_H, eigvec_H, Q

    def run_arc_cont(self, X, dX_dr, dr, X1):
        '''Iterates Newton-GMRes solver until convergence using arclength continuation
        Follows convention of Chandler - Kerswell: Invariant recurrent solutions.. 
        X: Vector containing (U, T, sx, lda) to be updated with Newton until it converges to periodic orbit
        dX_dr: Derivative of X w.r.t. arclength 
        X1: Previous converged solution (X(r0) in Chandler-Kerswell '''

        for iN in range(self.pm.restart+1, self.pm.N_newt):
            # lda: lambda parameter, Re (Reynolds) if solver=='KolmogorovFlow', Ra (Rayleigh) if solver=='BOUSS'
            U, T, sx, lda = self.unpack_X(X, arclength = True)

            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A_arc(X, dX_dr, iN)

            # RHS of Newton extended system
            # form b (rhs)
            b = self.form_b(U, UT)

            # add arclength term
            N = self.N_constraint(X, dX_dr, dr, X1)
            b = np.append(b, -N)

            F = self.norm(b[:-1]) # F only includes diff between U and UT, not arclength constraint

            # Write to txts
            self.write_prints(iN, F, U, sx, T, (lda, N))

            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, self.pm.N_gmres, self.pm.tol_gmres, iN, self.pm.glob_method)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN, arclength = True)

            # Update solution
            U, T, sx, lda = self.unpack_X(X, arclength = True)

            # Select different initial condition from orbit if solution is not converging
            if (1-F_new/F) < self.pm.tol_nudge:
                with open('prints/nudge.txt', 'a') as file:
                    file.write(f'iN = {iN}. frac_nudge = {self.pm.frac_nudge}\n')
                U = self.evolve(U, T*self.pm.frac_nudge)

            # Termination condition
            if F_new < self.pm.tol_newt:
                self.mkdirs_iN(iN)
                UT = self.evolve(U, T, save = True, iN = iN)
                if self.pm.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                # add arclength term
                N = self.N_constraint(X, dX_dr, dr, X1)
                b = np.append(b, -N)
                F = self.norm(b[:-1]) # F only includes diff between U and UT, not arclength constraint

                # Write to txts
                self.write_prints(iN+1, F, U, sx, T, (lda, N))
                break

    def update_lda(self, lda):
        '''Updates lambda parameter in solver'''
        if self.solver.solver == 'KolmogorovFlow':
            Re = lda # Reynolds number
            self.pm.nu = 1 / Re
        elif self.solver.solver == 'BOUSS': 
            Ra = lda # Rayleigh number
            self.pm.ra = Ra
        else:
            raise Exception('Arclength only implemented for Kolmogorov and BOUSS')

    def N_constraint(self, X, dX_dr, dr, X1):
        '''Calculates N function resulting from the arclength constraint as in 
        'Chandler - Kerswell: Invariant recurrent solutions..' '''
        return np.dot(dX_dr, (X-X1)) - dr

    def update_A_arc(self, X, dX_dr, iN):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, T, sx, lda = self.unpack_X(X, arclength = True)

        # Update lambda
        self.update_lda(lda)

        # Evolve fields and save output
        self.mkdirs_iN(iN-1) # save output and bal from previous iN
        UT = self.evolve(U, T, save = True, iN = iN-1)

        # Translate UT by sx and calculate derivatives
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

            dUT_ds = self.deriv_U(UT)
            dU_ds = self.deriv_U(U)
        else:
            dUT_ds = dU_ds = np.zeros_like(U) # No translation if sx is None

        # Calculate derivatives in time
        dUT_dT = self.evolve(UT, self.pm.dt)
        dUT_dT = (dUT_dT - UT)/self.pm.dt

        dU_dt = self.evolve(U, self.pm.dt)
        dU_dt = (dU_dt - U)/self.pm.dt

        # Calculate derivative of evolved translated fields wrt lambda
        dlda = lda * 1e-2
        self.update_lda(lda + dlda)
        dUT_dlda = self.evolve(U, T)
        dUT_dlda = self.translate(dUT_dlda,sx)
        dUT_dlda = (dUT_dlda - UT)/dlda

        # Return lambda to previous state
        self.update_lda(lda)

        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds, dlda = self.unpack_X(dX, arclength = True)

            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU and apply solenoidal projection if sp1 = True
            U_pert = self.apply_proj(U, epsilon*dU, self.pm.sp1)

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon

            # Calculate projections of dU needed for extended Newton system
            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real
            lda_proj = np.dot(dX_dr.conj(), dX).real

            # Save norms for diagnostics
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds, dUT_dlda]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj, lda_proj)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT + dUT_dlda*dlda
            if self.pm.sx is not None:
                return np.append(LHS, [t_proj, Tx_proj, lda_proj])
            else:
                return np.append(LHS, [t_proj, lda_proj])

        return apply_A, UT
