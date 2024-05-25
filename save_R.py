"""
- Se selecciona una simulación y se cargan los datos de outputs.
- Se usa una función Resto para comparar con norma L^2 entre 2 snapshots
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.animation as animation
import params2D as pm
import mod2D as mod

# Initialize solver
grid = mod.Grid(pm)


def get_var_name(variable):
    """devuelve el nombre de la variable (para usar en plt.title)
    """
    globals_dict = globals()

    return [
        var_name for var_name in globals_dict
        if globals_dict[var_name] is variable
    ][0] 

#defino función Resto: distancia en L^2 entre 2 tiempos
def R(V, t_idx, T_idx): #Toma el campo V, el tiempo t, y el período T. También se puede ingresar el t_idx directamente         
    Resto = np.linalg.norm(V[T_idx + t_idx,:,:]-V[t_idx,:,:])/np.linalg.norm(V[t_idx,:,:])
    return Resto    


#R2: considera 2 campos (Vx, Vz) para tomar el resto.
def R2(u, v, t_idx, T_idx): #Toma el campo v, el tiempo t, y el período T. También se puede ingresar el t_idx directamente         
    R_u = np.linalg.norm(u[T_idx + t_idx,:,:]-u[t_idx,:,:])
    R_v = np.linalg.norm(v[T_idx + t_idx,:,:]-v[t_idx,:,:])
    
    R_norm = np.sqrt((np.linalg.norm(u[t_idx,:,:]))**2 + (np.linalg.norm(v[t_idx,:,:]))**2)
    R_uv = np.sqrt(R_u**2 +R_v**2)/R_norm
    R_u = R_u/np.linalg.norm(u[t_idx,:,:])
    R_v = R_v/np.linalg.norm(v[t_idx,:,:])
    return R_uv, R_u, R_v    

def vort(u, v):
    fu, fv = mod.forward(u), mod.forward(v)
    uy = mod.inverse(mod.deriv(fu, grid.ky))
    vx = mod.inverse(mod.deriv(fv, grid.kx))
    return uy - vx

def R_vort_function(u, v, t_idx, T_idx): #Toma el campo v, el tiempo t, y el período T. También se puede ingresar el t_idx directamente         
    oz = vort(u[t_idx,:,:], v[t_idx,:,:])
    oz_T = vort(u[t_idx + T_idx,:,:], v[t_idx + T_idx,:,:])
    
    R_w = np.linalg.norm(oz_T-oz)

    return R_w/np.linalg.norm(oz)    


path = '/share/data4/jcullen/pySPEC/run1/'
fields = np.load(f'{path}output2/fields_t.npz')
uu_t = fields['uu_t']
vv_t = fields['vv_t']

dt = 1e-3
save_freq = 50
nx = ny = 256

start_t, end_t, start_T, end_T = 0,1448, 0.05, 50
# start_t, end_t, start_T, end_T = .1,10, .1, 10

start_T_idx = int((start_T/dt) /save_freq)
N_T = int(((end_T-start_T)/dt)/save_freq)

start_t_idx = int((start_t/dt) /save_freq)
N_t = int(((end_t-start_t)/dt)/save_freq)


####### Save R for u, v, uv: ##########

# R_uv = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t
# R_u = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t
# R_v = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t

# i = 0
# for t_idx in range(N_t):
#     if t_idx%(N_t/10) == 0:
#         i += 1
#         print(f'{i*10}% guardado')
#     for T_idx in range(N_T):
#         R_uv[t_idx, T_idx],R_u[t_idx, T_idx],R_v[t_idx, T_idx] = R2(uu_t, vv_t, t_idx + start_t_idx,T_idx+start_T_idx)

# np.save(f'{path}R/R2_uv.npy', R_uv)
# np.save(f'{path}R/R2_u.npy', R_u)
# np.save(f'{path}R/R2_v.npy', R_v)

####### Save R for vorticity w #########
step = 20

R_w = np.zeros((N_t//step, N_T//20)) #guardo los R de V para diferentes N_T dif T, y N_t dif t

i = 0
for idx, t_idx in enumerate(range(0,N_t-step,step)):
    if idx%int(N_t//step/100) == 0:
        i += 1
        print(f'{i}% guardado')
    for idx2, T_idx in enumerate(range(0,N_T-step,step)):
        if (idx>=R_w.shape[0]) or (idx2>R_w.shape[1]):
            print('index error')
            break
        R_w[idx, idx2] = R_vort_function(uu_t, vv_t, t_idx + start_t_idx,T_idx+start_T_idx)

np.save(f'{path}R/R_w.npy', R_w)
