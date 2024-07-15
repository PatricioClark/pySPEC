import numpy as np
import matplotlib.pyplot as plt
from gmres_kol import back_substitution
import os

class Grid:
    def __init__(self, pm):
        xi, dx = np.linspace(0, pm.Lx, pm.Nx, endpoint=False, retstep=True)
        yi, dy = np.linspace(0, pm.Ly, pm.Ny, endpoint=False, retstep=True)
        xx, yy = np.meshgrid(xi, yi) #should add indexing = 'ij' if Nx and Ny are different for clarity
        tt, dt = np.linspace(0, pm.T, pm.Nt, endpoint=False, retstep=True)

        kx = np.fft.fftfreq(pm.Nx, dx) 
        ky = np.fft.fftfreq(pm.Ny, dy) 
        kx, ky = np.meshgrid(kx, ky)
        k2 = kx**2 + ky**2
        kk = np.sqrt(k2)
        kr = np.round(kk)

        self.xx = xx
        self.dx = dx
        self.yy = yy
        self.dy = dy
        self.tt = tt
        self.dt = dt

        self.kx = kx
        self.ky = ky
        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.norm = 1.0/(pm.Nx*pm.Ny)**2

        # De-aliasing modes
        self.zero_mode = (0, 0)
        self.dealias_modes = (kk > pm.Nx/3)

        # Solenoidal mode proyector
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1.0 - kx**2/k2)
            self.pyy = np.nan_to_num(1.0 - ky**2/k2)
            self.pxy = np.nan_to_num(- kx*ky/k2)

def forward(ui):
    ''' Forward Fourier transform '''
    return np.fft.fft2(ui)

def inverse(ui):
    ''' Invserse Fourier transform '''
    return np.fft.ifft2(ui).real

def deriv(ui, ki):
    ''' First derivative in ki direction '''
    return 1.0j*ki*ui

def avg(ui, grid):
    ''' Mean in Fourier space '''
    return grid.norm * np.sum(ui)

def inner(a, b, c, d):
    ''' Inner product '''
    return (a*c.conjugate() + b*d.conjugate()).real

def inc_proj(fu, fv, grid):
    ''' Project onto solenoidal modes '''
    return grid.pxx*fu + grid.pxy*fv, grid.pxy*fu + grid.pyy*fv

# def check(fu, fv, grid, pm, step, nstep):
#     u2 = inner(fu, fv, fu, fv)

#     uy = inverse(deriv(fu, grid.ky))
#     vx = inverse(deriv(fv, grid.kx))
#     oz = uy - vx

#     uk = np.array([np.sum(u2[np.where(grid.kr == ki)]) for ki in range(pm.Nx//2)])
#     Ek = 0.5 * grid.norm * uk

#     np.save(f'Ek/nstep{nstep}/Ek.{step:06}', Ek)    
#     np.save(f'evol/nstep{nstep}/oz.{step:06}', oz)    


def vort(uu, vv, pm, grid):
    ''' Computes vorticity field '''
    fu, fv = forward(uu), forward(vv)
    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    if not pm.fvort:
        return uy - vx
    else:
        return forward(uy-vx)

def inv_vort(oz, pm, grid):
    ''' Computes uu,vv from oz vorticity'''
    if not pm.fvort:
        foz = forward(oz)
    else:
        foz = oz
    fu = np.divide(-deriv(foz, grid.ky), grid.k2, out = np.zeros_like(foz), where = grid.k2!=0.)
    fv = np.divide(deriv(foz, grid.kx), grid.k2, out = np.zeros_like(foz), where = grid.k2!=0.)
    uu, vv = inverse(fu), inverse(fv)
    return uu, vv

def flatten_fields(uu, vv, pm, grid):
    ''' Transforms uu, vv, to a 1d variable X (if vort saves oz)'''
    if not (pm.vort_X or pm.fvort):
        return np.concatenate((uu.flatten(), vv.flatten()))
    else:
        oz = vort(uu, vv, pm, grid)
        return oz.flatten()

def unflatten_fields(X, pm, grid):
    ''' Transforms 1d variable X to uu,vv '''
    if not (pm.vort_X or pm.fvort):
        ll = len(X)//2
        uu = X[:ll].reshape((pm.Nx, pm.Ny))
        vv = X[ll:].reshape((pm.Nx, pm.Ny))
        return uu, vv
    else:
        oz = X.reshape((pm.Nx, pm.Ny))
        uu, vv = inv_vort(oz, pm, grid)
        return uu, vv

def save_X(X, fname, pm, grid):
    ''' saves 1d variable X '''
    uu, vv = unflatten_fields(X, pm, grid)
    np.savez(f'output/fields_{fname}.npz', uu=uu,vv=vv)

def save_oz_Ek(oz_t, Ek_t, fu, fv, grid, pm, idx):
    ''' Save vorticity and spectrum '''
    #vorticity
    uy = inverse(deriv(fu, grid.ky))
    vx = inverse(deriv(fv, grid.kx))
    oz = uy - vx

    #spectrum
    u2 = inner(fu, fv, fu, fv)
    uk = np.array([np.sum(u2[np.where(grid.kr == ki)]) for ki in range(pm.Nx//2)])
    Ek = 0.5 * grid.norm * uk

    oz_t[idx, :,:] = oz
    Ek_t[idx, :] = Ek

def balance(fu, fv, fx, fy, grid, pm, i_newt, step):
    ''' Save global quantities: energy, dissipation, injection rate '''
    u2 = inner(fu, fv, fu, fv)
    eng = 0.5 * avg(u2, grid)
    dis = - pm.nu * avg(grid.k2*u2, grid)
    inj = avg(inner(fu, fv, fx, fy), grid)

    with open(f'balance/balance{i_newt}.dat','a') as ff:
        print(f'{step*pm.dt}, {eng}, {dis}, {inj}', file=ff) 


def evolution_function(grid, pm):
    # Forcing
    kf = 4
    fx = np.sin(2*np.pi*kf*grid.yy/pm.Lx)
    fx = forward(fx)
    fy = np.zeros((pm.Nx, pm.Nx), dtype=complex)
    fx, fy = inc_proj(fx, fy, grid)

    def evolve(X, T, save = False, i_newt = 0):
        ''' Evolves 1d vector X a time T. if save, saves balance and vort fields in i_newt directory'''
        uu, vv = unflatten_fields(X, pm, grid)
        fu = forward(uu)
        fv = forward(vv)

        Nt = round(T/pm.dt)
        if save: #save vorticity and Ek every ffreq
            oz_t = np.empty(((Nt//pm.ffreq)+1, pm.Nx, pm.Ny))
            Ek_t = np.empty(((Nt//pm.ffreq)+1, pm.Nx//2))

        for step in range(Nt):
            
            if save:
                #save fields and Ek
                if step% pm.ffreq==0:
                    save_oz_Ek(oz_t, Ek_t, fu, fv, grid, pm, idx = step//pm.ffreq)
                #save balance
                if step% pm.bfreq==0:
                    balance(fu, fv, fx, fy, grid, pm, i_newt, step)

            # Store previous time step
            fup = np.copy(fu)
            fvp = np.copy(fv)
       
            # Time integration
            for oo in range(pm.rkord, 0, -1):
                # Non-linear term
                uu = inverse(fu)
                vv = inverse(fv)
                
                ux = inverse(deriv(fu, grid.kx))
                uy = inverse(deriv(fu, grid.ky))

                vx = inverse(deriv(fv, grid.kx))
                vy = inverse(deriv(fv, grid.ky))

                gx = forward(uu*ux + vv*uy)
                gy = forward(uu*vx + vv*vy)
                gx, gy = inc_proj(gx, gy, grid)

                # Equations
                fu = fup + (grid.dt/oo) * (
                    - gx
                    - pm.nu * grid.k2 * fu 
                    + fx
                    )

                fv = fvp + (grid.dt/oo) * (
                    - gy
                    - pm.nu * grid.k2 * fv 
                    + fy
                    )

                # de-aliasing
                fu[grid.zero_mode] = 0.0 
                fv[grid.zero_mode] = 0.0 
                fu[grid.dealias_modes] = 0.0 
                fv[grid.dealias_modes] = 0.0

        if save:
            #save last field and Ek
            save_oz_Ek(oz_t, Ek_t, fu, fv, grid, pm, idx = -1 )
            np.save(f'evol/oz_t{i_newt}', oz_t)    
            np.save(f'Ek/Ek_t{i_newt}', oz_t)    
            #save last balance
            balance(fu, fv, fx, fy, grid, pm, i_newt, step = Nt)
                    
        uu = inverse(fu)
        vv = inverse(fv)
        X  = flatten_fields(uu, vv, pm, grid)
        return X
    return evolve


def inc_proj_X_function(grid, pm):
    def inc_proj_X(X):
        ''' Projects X to solenoidal modes '''
        uu, vv = unflatten_fields(X, pm, grid)
        fu = forward(uu)
        fv = forward(vv)
        fu, fv = inc_proj(fu, fv, grid)
        uu = inverse(fu)
        vv = inverse(fv)
        X  = flatten_fields(uu, vv, pm, grid)
        return X
    return inc_proj_X

def inf_trans_function(grid, pm):
    def inf_trans(X):
        ''' Performs derivative of X in x direction '''
        uu, vv = unflatten_fields(X, pm, grid)
        fu = forward(uu)
        fv = forward(vv)
        fu = deriv(fu, grid.kx)
        fv = deriv(fv, grid.kx)

        uu = inverse(fu)
        vv = inverse(fv)
        X  = flatten_fields(uu, vv, pm, grid)
        return X
    return inf_trans

def translation_function(grid, pm):
    def translation(X, s):
        ''' Performs finite translation '''
        uu, vv = unflatten_fields(X, pm, grid)
        fu = forward(uu)
        fv = forward(vv)
        fu = fu * np.exp(1.0j*grid.kx*s)
        fv = fv * np.exp(1.0j*grid.kx*s)

        uu = inverse(fu)
        vv = inverse(fv)
        X  = flatten_fields(uu, vv, pm, grid)
        return X
    return translation

def application_function(evolve, inf_trans, translation, pm, X, T, Y, s, i_newt):
    ''' Updates matrix A (from newton method) for every newton iteration'''
    #compute variables and derivatives used throughout gmres iterations
    norm_X = np.linalg.norm(X)
    Tx_Y = inf_trans(Y)
    Tx_X = inf_trans(X)

    dY_dT = evolve(Y, pm.dt) - Y
    dY_dT = dY_dT/pm.dt

    dX_dt = evolve(X, pm.dt) - X
    dX_dt = dX_dt/pm.dt

    def apply_A(dX, ds, dT):
        ''' Applies A matrix to vector (dX,ds,dT)^t  '''        
        norm_dX = np.linalg.norm(dX)
        epsilon = 1e-7*norm_X/norm_dX

        deriv_x0 = evolve(X+epsilon*dX, T)
        deriv_x0 = translation(deriv_x0,s) - Y
        deriv_x0 = deriv_x0/epsilon

        t_proj = np.dot(dX_dt.conj(), dX).real
        Tx_proj = np.dot(Tx_X.conj(), dX).real

        with open(f'prints/apply_A/iter{i_newt}.txt', 'a') as file:
            file.write(f'{round(norm_dX,4)},{round(np.linalg.norm(deriv_x0),4)},{round(np.linalg.norm(dX_dt),4)},\
            {round(np.linalg.norm(dY_dT),4)},{round(t_proj,4)},{round(Tx_proj,4)}\n')

        LHS = deriv_x0-dX+Tx_Y*ds+dY_dT*dT
        return np.append(LHS, [Tx_proj, t_proj])
    return apply_A

def trust_region(Delta, H, beta, k, Q, pm, i_newt):
    ''' Applies trust region to solution provided by GMRes '''
    R = H[:k,:k]
    y = back_substitution(R, beta[:k], pm) #computes initial solution by GMRes
    mu = 0.
    y_norm = np.linalg.norm(y)

    with open(f'prints/hookstep/extra_iter{i_newt}.txt', 'a') as file:
        file.write(f'{Delta},{mu},{np.linalg.norm(y)},0\n')

    for j in range(100): #100 big enough to ensure that condition is satisfied
        if y_norm <= Delta:
            break
        else:
            mu = 1e-3 * 2.**j
            A = R.T @ R + mu * np.eye(R.shape[0])
            b = R.T @ beta[:k]
            y = np.linalg.solve(A, b) #updates solution with trust region
            y_norm = np.linalg.norm(y)
            with open(f'prints/hookstep/extra_iter{i_newt}.txt', 'a') as file:
                file.write(f'{Delta},{mu},{np.linalg.norm(y)},{round(np.linalg.cond(A, 2))}\n')
            
    x = Q[:,:k]@y
    return x[:-2], x[-2], x[-1]


def hookstep_function(inc_proj_X, evolve, translation, pm):
    def hookstep(H, beta, Q, k, X, sx, T, b, i_newt):
        ''' Performs hookstep on solution given by GMRes untill new |b| is less than previous |b| (or max iter of hookstep is reached) '''
        b_norm = np.linalg.norm(b)
        y = back_substitution(H[:k,:k], beta[:k], pm)
        Delta = np.linalg.norm(y)
        
        for i_hook in range(pm.N_hook):
            dX, dsx, dT = trust_region(Delta, H, beta, k, Q, pm, i_newt)
            dX = inc_proj_X(dX)
            X_new = X+dX
            sx_new = sx+dsx.real
            T_new = T+dT.real

            Y = evolve(X_new, T_new)
            Y = translation(Y,sx_new)
            b_new = X_new-Y
            b_new = np.append(b_new, [0., 0.])
            
            with open(f'prints/hookstep/iter{i_newt}.txt', 'a') as file:
                file.write(f'{i_hook+1},{round(np.linalg.norm(b_new),5)}\n')
            
            if np.linalg.norm(b_new)<= b_norm:
                break
            else:
                Delta *= .5 #reduce trust region
        return X_new, Y, sx_new, T_new, b_new 
    return hookstep


def line_search_function(inc_proj_X, evolve, translation, pm):
    def linesearch(H, beta, Q, k, X, sx, T, b, i_newt):
        ''' Performs line search on solution given by GMRes untill new |b| is less than previous |b| (or max iter of linesearch is reached) '''
        b_norm = np.linalg.norm(b)
        y = back_substitution(H[:k,:k], beta[:k], pm)
        x = Q[:,:k]@y

        for i_hook in range(pm.N_hook):
            dX, dsx, dT = x[:-2], x[-2], x[-1]
            dX = inc_proj_X(dX)
            X_new = X+dX
            sx_new = sx+dsx.real
            T_new = T+dT.real

            Y = evolve(X_new, T_new)
            Y = translation(Y,sx_new)
            b_new = X_new-Y
            b_new = np.append(b_new, [0., 0.])
            
            with open(f'prints/linesearch/iter{i_newt}.txt', 'a') as file:
                file.write(f'{i_hook+1},{round(np.linalg.norm(b_new),5)}\n')
            
            if np.linalg.norm(b_new)<= b_norm:
                break
            else:
                x *= .5 #reduce trust region by changing step size
        return X_new, Y, sx_new, T_new, b_new 
    return linesearch

def write_prints(i_newt, b_norm, X, sx, T):
    with open('prints/b_error.txt', 'a') as file1,\
    open('prints/solver.txt', 'a') as file2, \
    open(f'prints/error_gmres/iter{i_newt}.txt', 'w') as file3,\
    open(f'prints/hookstep/iter{i_newt}.txt', 'w') as file4,\
    open(f'prints/apply_A/iter{i_newt}.txt', 'w') as file5,\
    open(f'prints/linesearch/iter{i_newt}.txt', 'w') as file6,\
    open(f'prints/hookstep/extra_iter{i_newt}.txt', 'w') as file7:
        file1.write(f'{i_newt-1},{round(b_norm,3)}\n')
        file2.write(f'{i_newt-1},{round(T, 8)},{round(sx, 8)},{round(np.linalg.norm(X), 3)}\n')
        file3.write('iter gmres, error\n')
        file4.write('iter hookstep, |b|\n')
        file5.write('|dX|,|dY/dX|,|dX/dt|,|dY/dT|,t_proj,Tx_proj\n')
        file6.write('iter linesearch, |b|\n')
        file7.write('Delta, mu, |y|, cond(A)\n')

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

def mkdirs():
    mkdir('output')
    mkdir('evol')
    mkdir('Ek')
    mkdir('balance')
    mkdir('prints')

    for print_dir in ('error_gmres', 'hookstep','linesearch','apply_A'):
        mkdir(f'prints/{print_dir}')


#Possibility: encompass hook, ls, and no TR approach in one function:
# def post_GMRES_function(grid, pm):

#     def post_GMRES(H, beta, Q, k, X, sx, T, b, i_newt):
#         if pm.hook:        

#     return post_GMRES    


#Previous faulty version: Hookstep transformation was miscalculated
# def trust_region(Delta, H, beta, k, Q, pm):
#     ''' Applies trust region to solution provided by GMRes '''
#     y = back_substitution(H[:k,:k], beta[:k], pm) #computes initial solution by GMRes
#     mu = 0.

#     for j in range(100): #100 big enough to ensure that condition is satisfied
#         y_norm = np.linalg.norm(y)
#         if y_norm <= Delta:
#             break
#         else:
#             mu = 2.**j
#             H = H_transform(H, mu, k)
#             y = back_substitution(H[:k,:k], beta[:k], pm) #updates solution with trust region
    
#     x = Q[:,:k]@y
#     return x[:-2], x[-2], x[-1]

# def H_transform(H, mu, k):
#     ''' Performs trust region transform to GMRes Hessenberg matrix H '''
#     for i in range(k):
#         H[i,i] += mu/H[i,i] 
#     return H

# def hookstep_function(inc_proj_X, evolve, translation, pm):
#     def hookstep(H, beta, Q, k, X, sx, T, b, i_newt):
#         ''' Performs hookstep on solution given by GMRes untill new |b| is less than previous |b| (or max iter of hookstep is reached) '''
#         b_norm = np.linalg.norm(b)
#         y = back_substitution(H[:k,:k], beta[:k], pm)
#         Delta = np.linalg.norm(y)
        
#         for i_hook in range(pm.N_hook):
#             dX, dsx, dT = trust_region(Delta, H, beta, k, Q, pm)
#             dX = inc_proj_X(dX)
#             X_new = X+dX
#             sx_new = sx+dsx.real
#             T_new = T+dT.real

#             Y = evolve(X_new, T_new)
#             Y = translation(Y,sx_new)
#             b_new = X_new-Y
#             b_new = np.append(b_new, [0., 0.])
            
#             with open(f'prints/hookstep/iter{i_newt}.txt', 'a') as file:
#                 file.write(f'{i_hook+1},{round(np.linalg.norm(b_new),5)}\n')
            
#             if np.linalg.norm(b_new)<= b_norm:
#                 break
#             else:
#                 Delta *= .5 #reduce trust region
#         return X_new, Y, sx_new, T_new, b_new 
#     return hookstep
