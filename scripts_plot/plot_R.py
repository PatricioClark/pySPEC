# Para hacer plots de R en mi compu:
import numpy as np
import matplotlib.pyplot as plt

Rs = ['R_u', 'R_v', 'R_uv']
start_t, end_t, start_T, end_T = 0,449, .05, 50

##Plot individual:
for R in Rs: 
    path = 'C:/Users/joaco/Desktop/FCEN/Tesis/Patrick/pySPEC/R/'
    R_ = np.load(path+R+'.npy')
    plt.figure(layout = 'constrained')#figsize = (14,6)
    plt.imshow(R_.T, origin = 'lower',aspect = 'auto', extent = (start_t,end_t, start_T,end_T))#, vmin = 0.1, vmax = 0.4)
    # plt.ylim(0.5,T)
    # plt.xlim(35,71-T)
    plt.xlabel('Tiempo inicial t')
    plt.ylabel(r'Tiempo $\tau$')
    plt.title(fr'${R}(t,\tau)$')
    plt.colorbar(orientation = 'horizontal', label = 'R')
    # plt.legend()
    plt.savefig(f'{path}{R}.png',dpi = 300, format = 'png')
plt.show()

