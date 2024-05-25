# Para hacer plots de R en mi compu:
import numpy as np
import matplotlib.pyplot as plt

output = 2

if output == 2:
    Rs = ['R2_u', 'R2_v', 'R2_uv',  'R_w']
    labels = ['R_u', 'R_v', 'R_{uv}', 'R_\omega']
    start_t, end_t, start_T, end_T = 500,1948, .05, 50
elif output == 3:
    Rs = ['R3_w']
    labels = ['R_\omega']
    start_t, end_t, start_T, end_T = 2000,3900, .05, 50

#Output==1
# start_t, end_t, start_T, end_T = 0,448, .05, 50

##Plot individual:
path = 'C:/Users/joaco/Desktop/FCEN/Tesis/Patrick/pySPEC/R/'
for R, label in zip(Rs, labels): 
    R_ = np.load(path+R+'.npy')

    plt.figure(layout = 'constrained')#figsize = (14,6)
    plt.imshow(R_.T, origin = 'lower',aspect = 'auto', extent = (start_t,end_t, start_T,end_T))#, vmin = 0.1, vmax = 0.4)
    plt.scatter(897.8, 22.7, label = 'guessed orbit')
    plt.xlabel('Tiempo inicial t')
    plt.ylabel(r'Tiempo $\tau$')
    plt.title(fr'${label}(t,\tau)$')
    plt.colorbar(orientation = 'horizontal', label = 'R')
    # plt.legend()
    # plt.savefig(f'{path}{R}.png',dpi = 300, format = 'png')
plt.show()

# for R, label in zip(Rs, labels): 

#     R_ = np.load(path+R+'.npy')
#     ti = np.linspace(start_t, end_t, R_.shape[0], endpoint = True)
#     Ti = np.linspace(start_T, end_T, R_.shape[1], endpoint = True)
#     tt, TT = np.meshgrid(ti, Ti)

#     plt.figure(layout = 'constrained')#figsize = (14,6)
#     plt.contourf(tt, TT, R_.T)
#     plt.xlabel('Tiempo inicial t')
#     plt.ylabel(r'Tiempo $\tau$')
#     plt.title(fr'${label}(t,\tau)$')
#     plt.colorbar(orientation = 'horizontal', label = 'R')
#     # plt.legend()
#     # plt.savefig(f'{path}{R}.png',dpi = 300, format = 'png')
# plt.show()
