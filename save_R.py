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
def R2(V1, V2, t_idx, T_idx): #Toma el campo V, el tiempo t, y el período T. También se puede ingresar el t_idx directamente         
    R_V1 = np.linalg.norm(V1[T_idx + t_idx,:,:]-V1[t_idx,:,:])
    R_V2 = np.linalg.norm(V2[T_idx + t_idx,:,:]-V2[t_idx,:,:])
    
    R_norm = np.sqrt((np.linalg.norm(V1[t_idx,:,:]))**2 + (np.linalg.norm(V2[t_idx,:,:]))**2)
    Resto = np.sqrt(R_V1**2 +R_V2**2)/R_norm
    return Resto    


T = 400
dt = 1e-3
save_freq = 50
nx = ny = 256
Nt = int(T/dt //save_freq)

uu_t = np.zeros((Nt,nx,ny))
vv_t = np.zeros((Nt,nx,ny))

path = '/share/data4/jcullen/pySPEC/run1/'
for i in range(Nt):
    step = save_freq * (1 + i)
    f_name = f'fields_{step:06}'
    fields = np.load(f'{path}output/{f_name}.npz')
    uu = fields['uu']
    vv = fields['vv']
    uu_t[i,:,:] = uu
    vv_t[i,:,:] = vv
    if i%1000 == 0:
        print('saved 1000 more')

print('Saved fields')


def save_R(V, start_t, end_t, start_T, end_T):
    """
    Args:
	V: Campo vectorial 2D con t en primer col.
	t: tiempo inicial de snapshot
	T: guess de periodo. Compara V(t) con V(t+T)
    Returns:
	R(t, T). Función escalar que devuelve la resta en norma L^2 entre los estados a tiempos t y t+T
    """
    start_T_idx = int((start_T/dt) /save_freq)
    N_T = int(((end_T-start_T)/dt)/save_freq)

    start_t_idx = int((start_t/dt) /save_freq)
    N_t = int(((end_t-start_t)/dt)/save_freq)
    
    R_V = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t

    for T_idx in range(1,N_T):
        for t_idx in range(1, N_t):
            R_V[t_idx, T_idx] = R(V, t_idx + start_t_idx,T_idx+start_T_idx)

    np.save(path+'R/R_{get_var_name(V)}_T20.npy', R_V)
    print('Saved ', get_var_name(V))


# save_R(Vz, start_t= 0, end_t = 279, start_T = 0, end_T = 20)
# save_R(Vx, start_t= 0, end_t = 279, start_T = 0, end_T = 20)
# save_R(Th, start_t= 0, end_t = 279, start_T = 0, end_T = 20)


# #Para V2 lo hago aparte porque usa otra función:
start_t, end_t, start_T, end_T = .1,300, .1, 50
# start_t, end_t, start_T, end_T = .1,10, .1, 10

start_T_idx = int((start_T/dt) /save_freq)
N_T = int(((end_T-start_T)/dt)/save_freq)

start_t_idx = int((start_t/dt) /save_freq)
N_t = int(((end_t-start_t)/dt)/save_freq)

R_V = np.zeros((N_t, N_T)) #guardo los R de V para diferentes N_T dif T, y N_t dif t

for T_idx in range(N_T):
    for t_idx in range(N_t):
        R_V[t_idx, T_idx] = R2(uu_t, vv_t, t_idx + start_t_idx,T_idx+start_T_idx)

np.save(f'{path}R/R_V2.npy', R_V)

###Plot individual:
plt.figure()
plt.figure(layout = 'constrained')#figsize = (14,6)
plt.imshow(R_V, origin = 'lower',aspect = 'auto', extent = (start_t,end_t, start_T,end_T))#, vmin = 0.1, vmax = 0.4)
# plt.ylim(0.5,T)
# plt.xlim(35,71-T)
plt.xlabel('Tiempo inicial t')
plt.ylabel(r'Tiempo $\tau$')
plt.title(fr'$R(t,\tau)$')
plt.colorbar(orientation = 'horizontal', label = 'R')
# plt.legend()
plt.savefig(f'{path}R/R_V2.png',dpi = 300, format = 'png')
plt.show()
