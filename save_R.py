"""
- Se selecciona una simulación y se cargan los datos de outputs.
- Se usa una función Resto para comparar con norma L^2 entre 2 snapshots
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.animation as animation


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

path = '/share/data4/jcullen/pySPEC/run1/'
fields = np.load(f'{path}output/fields_t.npz')
uu_t = fields['uu_t']
vv_t = fields['vv_t']

# T = 400
dt = 1e-3
save_freq = 50
nx = ny = 256
# Nt = int(T/dt //save_freq)


# def save_R(V, start_t, end_t, start_T, end_T):
#     """
#     Args:
# 	V: Campo vectorial 2D con t en primer col.
# 	t: tiempo inicial de snapshot
# 	T: guess de periodo. Compara V(t) con V(t+T)
#     Returns:
# 	R(t, T). Función escalar que devuelve la resta en norma L^2 entre los estados a tiempos t y t+T
#     """
#     start_T_idx = int((start_T/dt) /save_freq)
#     N_T = int(((end_T-start_T)/dt)/save_freq)

#     start_t_idx = int((start_t/dt) /save_freq)
#     N_t = int(((end_t-start_t)/dt)/save_freq)
    
#     R_V = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t

#     for T_idx in range(1,N_T):
#         for t_idx in range(1, N_t):
#             R_V[t_idx, T_idx] = R(V, t_idx + start_t_idx,T_idx+start_T_idx)

#     np.save(path+'R/R_{get_var_name(V)}.npy', R_V)
#     print('Saved ', get_var_name(V))


# save_R(Vz, start_t= 0, end_t = 279, start_T = 0, end_T = 20)
# save_R(Vx, start_t= 0, end_t = 279, start_T = 0, end_T = 20)
# save_R(Th, start_t= 0, end_t = 279, start_T = 0, end_T = 20)


#Para V2 lo hago aparte porque usa otra función:
start_t, end_t, start_T, end_T = 0,449, 0.05, 50
# start_t, end_t, start_T, end_T = .1,10, .1, 10

start_T_idx = int((start_T/dt) /save_freq)
N_T = int(((end_T-start_T)/dt)/save_freq)

start_t_idx = int((start_t/dt) /save_freq)
N_t = int(((end_t-start_t)/dt)/save_freq)

R_uv = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t
R_u = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t
R_v = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t

i = 0
for t_idx in range(N_t):
    if t_idx%(N_t/10) == 0:
        i += 1
        print(f'{i*10}% guardado')
    for T_idx in range(N_T):
        R_uv[t_idx, T_idx],R_u[t_idx, T_idx],R_v[t_idx, T_idx] = R2(uu_t, vv_t, t_idx + start_t_idx,T_idx+start_T_idx)
np.save(f'{path}R/R_uv.npy', R_uv)
np.save(f'{path}R/R_u.npy', R_u)
np.save(f'{path}R/R_v.npy', R_v)

# #%%
#Para hacer plots de R en mi compu:
# import numpy as np
# import matplotlib.pyplot as plt

# Rs = ['R_u', 'R_v', 'R_uv']
# start_t, end_t, start_T, end_T = 0,449, .05, 50

# ##Plot individual:
# for R in Rs: 
#     path = 'C:/Users/joaco/Desktop/FCEN/Tesis/Patrick/pySPEC/R/'
#     R_ = np.load(path+R+'.npy')
#     plt.figure(layout = 'constrained')#figsize = (14,6)
#     plt.imshow(R_.T, origin = 'lower',aspect = 'auto', extent = (start_t,end_t, start_T,end_T))#, vmin = 0.1, vmax = 0.4)
#     # plt.ylim(0.5,T)
#     # plt.xlim(35,71-T)
#     plt.xlabel('Tiempo inicial t')
#     plt.ylabel(r'Tiempo $\tau$')
#     plt.title(fr'${R}(t,\tau)$')
#     plt.colorbar(orientation = 'horizontal', label = 'R')
#     # plt.legend()
#     plt.savefig(f'{path}{R}.png',dpi = 300, format = 'png')
# plt.show()

