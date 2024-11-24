from .newton import NewtonSolver
from .gmres import GMRES, backsub
import os
import numpy as np

class UPO(NewtonSolver):
    def __init__(self, pm, pmN, solver):
        super().__init__(pmN, solver)
        self.pm = pm
        self.grid = solver.grid

    def mkdir(self, path):
        os.makedirs(path, exist_ok=True)

    def mkdirs(self):
        dirs = ['output', 'balance', 'prints/error_gmres', 'prints/hookstep', 'prints/apply_A'] # 'bin_temp' for wrapper
        for dir_ in dirs:
            self.mkdir(dir_)

    def mkdirs_iN(self, iN):
        dirs = [f'output/iN{iN:02}', f'balance/iN{iN:02}']
        for dir_ in dirs:
            self.mkdir(dir_)

    def load_ic(self):
        if not self.pmN.restart:
            # Start Newton Solver from initial guess
            fields = self.solver.load_fields(path = self.pmN.input, step = self.pmN.start_idx)
            T, sx = self.pmN.T_guess, self.pmN.sx_guess
            # Create directories
            self.mkdirs()
            self.write_header()
        else:
            # Restart Newton Solver from last iteration
            restart_path = f'output/iN{self.pmN.restart:02}/'
            fields = self.solver.load_fields(path = restart_path, step = 0)
            T, sx = self.get_restart_values(self.pmN.restart)
        return fields, sx, T

    def flatten(self, fields):
        return np.concatenate([f.flatten() for f in fields])

    def unflatten(self, U):
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
        if save:
            dir_iN = f'iN{iN:02}/'
            return self.solver.evolve(U, T, self.pm.bstep, self.pm.ostep, bpath = f'balance/{dir_iN}', opath = f'output/{dir_iN}')
        else:
            return self.solver.evolve(U, T)

    @flatten_dec
    def translate(self, U, sx):
        return self.grid.translate(U, sx)

    @flatten_dec
    def deriv_U(self, U):
        return self.grid.deriv_fields(U)

    @flatten_dec
    def sol_project(self, U):
        f = [self.grid.forward(ff) for ff in U]
        inc_f = self.solver.grid.inc_proj(f)         
        U = [self.grid.inverse(ff) for ff in inc_f]
        return U

    def form_b(self, U, UT):
        "Form RHS of extended Newton system. UT is evolved and translated flattened field"
        b = U - UT
        return np.append(b, [0., 0.])

    def norm(self, U):
        return np.linalg.norm(U)

    def get_restart_values(self, restart):
        solver = np.loadtxt('prints/solver.txt', delimiter = ',', skiprows = 1)
        idx_restart = np.argwhere(solver[:,0]==restart)[0][0] #in case more than 1 restart is needed find row with last iter
        T, sx = solver[idx_restart,1:3]
        return T, sx

    def update_A(self, X, iN):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, sx, T = X[:-2], X[-2], X[-1]

        # Evolve fields and save output
        self.mkdirs_iN(iN-1) # save output and bal from previous iN
        UT = self.evolve(U, T, save = True, iN = iN-1)
        UT = self.translate(UT, sx)

        dUT_ds = self.deriv_U(UT)
        dU_ds = self.deriv_U(U)

        dUT_dT = self.evolve(UT, self.pm.dt)
        dUT_dT = (dUT_dT - UT)/self.pm.dt

        dU_dt = self.evolve(U, self.pm.dt)
        dU_dt = (dU_dt - U)/self.pm.dt

        def apply_A(dU, ds, dT):
            ''' Applies A matrix to vector (dU,ds,dT)^t  '''        
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            U_pert = U + epsilon*dU
            with open ('test.txt', 'w') as file:
                file.write(f'{self.norm(U_pert)}\n')

            if self.pmN.sol_project:
                U_pert = self.sol_project(U_pert)
            with open ('test.txt', 'w') as file:
                file.write(f'{self.norm(U_pert)}\n')

            dUT_dU = self.evolve(U_pert, T)
            dUT_dU = (self.translate(dUT_dU,sx) - UT)/epsilon

            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real

            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds, ]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT
            return np.append(LHS, [Tx_proj, t_proj])

        return apply_A, UT

    def write_header(self):
        with open('prints/solver.txt', 'a') as file:
            file.write('Iter newt, |F|, T, sx, |U|\n')

    def write_prints(self, iN, F, U, sx, T):
        with open('prints/solver.txt', 'a') as file1,\
        open(f'prints/error_gmres/iN{iN:02}.txt', 'w') as file2,\
        open(f'prints/hookstep/iN{iN:02}.txt', 'w') as file3,\
        open(f'prints/apply_A/iN{iN:02}.txt', 'w') as file4,\
        open(f'prints/hookstep/extra_iN{iN:02}.txt', 'w') as file5:

            file1.write(f'{iN-1:02},{F:.4e},{T:.6e},{sx:.5e},{self.norm(U):.4e}\n')
            file2.write('iG, error\n')
            file3.write('iH, |F|, |F(x)+cAdx|, |F(x)+Adx|\n')
            file4.write('|U|, |dU|, |dUT/dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|, t_proj, Tx_proj\n')
            file5.write('Delta, mu, |y|, cond(A)\n')

    def write_apply_A(self, iN, norms, t_proj, Tx_proj):
        with open(f'prints/apply_A/iN{iN:02}.txt', 'a') as file:
            file.write(','.join([f'{norm:.4e}' for norm in norms]) + f',{t_proj:.4e},{Tx_proj:.4e}\n')

    def write_trust_region(self, iN, Delta, mu, y, A):
        with open(f'prints/hookstep/extra_iN{iN:02}.txt', 'a') as file:
            file.write(f'{Delta:.4e},{mu:.4e},{self.norm(y):.4e},{np.linalg.cond(A):.3e}\n')

    def write_hookstep(self, iN, iH, F_new, lin_exp, lin_exp_test):
        with open(f'prints/hookstep/iN{iN:02}.txt', 'a') as file:
            file.write(f'{iH:02},{F_new:.4e},{lin_exp:.4e},{lin_exp_test:.4e}\n')

    def iterate(self, fields, sx, T):            
        # Flatten fields
        U = self.flatten(fields)
        X = np.append(U, [sx, T])

        for iN in range(self.pmN.restart+1, self.pmN.N_newt):
            # Unpack X
            U, sx, T = X[:-2], X[-2], X[-1]
            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A(X, iN)
            
            # RHS of Newton extended system
            b = self.form_b(U, UT)
            # ||b|| = ||F||: rootsearch function
            F = self.norm(b)

            # Write to txts
            self.write_prints(iN, F, U, sx, T)
            
            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, iN, self.pmN)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN)
            # Update solution
            U, sx, T = X[:-2], X[-2], X[-1]

            # Termination condition
            if F_new < self.pmN.tol_newt:
                self.mkdirs_iN(iN)
                UT = self.evolve(U, T, save = True, iN = iN)
                break

    def hookstep(self, X, H, beta, Q, iN):
        ''' Performs hookstep on solution given by GMRes untill new |b| is less than previous |b| (or max iter of hookstep is reached) '''
        # Unpack X
        U, sx, T = X[:-2], X[-2], X[-1]
        #Initial solution from GMRes in basis Q
        y = backsub(H, beta, self.pmN)
        #Initial trust region radius
        Delta = self.norm(y)

        #Define trust_region function
        trust_region = self.trust_region_function(H, beta, iN, y)

        mu = 0.
        #Perform hookstep
        for iH in range(self.pmN.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y #Unitary transform back to full dimension
            dU, dsx, dT = dx[:-2], dx[-2], dx[-1] 

            #if projecting U+dU
            U_new = U+dU
            if self.pmN.sol_project:
                U_new = self.sol_project(U_new)
            
            sx_new = sx+dsx.real
            T_new = T+dT.real
            X_new = np.append(U_new, [sx_new, T_new])

            UT = self.evolve(U_new, T_new)
            UT = self.translate(UT,sx_new)
            F_new = self.norm(U_new-UT)
            
            lin_exp = self.norm(beta - self.pmN.c * H @ y) #linear expansion of F around x (in basis Q). 
            lin_exp_test = self.norm(beta - 1. * H @ y) #Test for checking if norm is equal to tol_gmres
            #F(x) + A dx (A is not the jacobian because of 2 extra conditions)

            self.write_hookstep(iN, iH, F_new, lin_exp, lin_exp_test)

            if F_new <= lin_exp:
                break
            else:
                Delta *= self.pmN.reduc_reg #reduce trust region
        return X_new, F_new, UT  


    def trust_region_function(self, H, beta, iN, y0):
        R = H #R part of QR decomposition
        A_ = R.T @ R #A matrix in trust region
        b = R.T @ beta #b vector in trust region

        def trust_region(Delta, mu):
            ''' Applies trust region to solution provided by GMRes '''
            ''' Delta: trust region radius, mu: penalty parameter  '''

            if mu == 0: #First hoostep iteration. Doens't perform trust region
                y_norm = self.norm(y0) 
                Delta0 = y_norm #Initial trust region

                with open(f'prints/hookstep/extra_iN{iN:02}.txt', 'a') as file:
                    file.write(f'{Delta0},{mu},{y_norm},0\n') 

                mu = self.pmN.mu0 #Initialize first nonzero value of mu
                return y0, mu

            for j in range(1, 1000): #1000 big enough to ensure that condition is satisfied

                # Ridge regression adjustment 
                A = A_ + mu * np.eye(A_.shape[0])
                y = np.linalg.solve(A, b) #updates solution with trust region

                self.write_trust_region(iN, Delta, mu, y, A)

                if self.norm(y) <= Delta:
                    break
                else:
                    # Increase mu until trust region is satisfied
                    mu *= 1.2 **j
                    
            return y, mu
        return trust_region
