# Para hacer plots de R en mi compu:
import numpy as np
import matplotlib.pyplot as plt

# path = 'C:/Users/joaco/Desktop/FCEN/Tesis/Patrick/pySPEC/R/'
path = 'C:/Users/joaco/Desktop/FCEN/Tesis/Patrick/Outputs/output_run1_1/R/'

#import R (takes t and T and returns norm of relative difference)
# R2_uv = np.load(path+'R2_uv'+'.npy')
# R2_w = np.load(path+'R2_w'+'.npy')
# R3_w = np.load(path+'R3_w'+'.npy')
 
R_uv = np.load(path+'R_V2_T20'+'.npy')


def plot_R(R, label, start_t,end_t, start_T,end_T):
    plt.figure(layout = 'constrained')#figsize = (14,6)
    plt.imshow(R.T, origin = 'lower',aspect = 'auto', extent = (start_t,end_t, start_T,end_T), vmin = 0.1, vmax = 0.4)
    # plt.scatter(897.8, 22.7, label = 'guessed orbit')
    plt.xlabel('Tiempo inicial t')
    plt.ylabel(r'Tiempo $\tau$')
    plt.title(fr'${label}(t,\tau)$')
    plt.colorbar(orientation = 'horizontal', label = 'R')
    # plt.savefig(f'{path}{R}.png',dpi = 300, format = 'png')
    
with open(f'{path}orbs.txt', 'x') as file:
    file.write(f'i, T, t, t_i, R_uv\n') #, R_w

#PySPEC:
# start_t, end_t, start_T, end_T = 500,1948, .05, 50

#SPECTER
T = 20
start_t, end_t, start_T, end_T = (0,299-T, 0,T)
plot_R(R_uv, 'R_{uv}', start_t, end_t, start_T, end_T)
#Save local minimums as guesses for newton solver 
tT_uv = plt.ginput(n=0, timeout=0, mouse_add=None, mouse_pop=None, mouse_stop=None)
tT_uv = np.array(tT_uv)
tT_uv = tT_uv[tT_uv[:,1].argsort()]

#Save R values for the guesses
Ruv_tT = np.zeros(len(tT_uv))
# Rw_tT = np.zeros(len(tT_uv))
t_idx = np.zeros(len(tT_uv))

for i, tT in enumerate(tT_uv):
    step_t = R_uv.shape[0]/(end_t-start_t)
    step_T = R_uv.shape[1]/(end_T-start_T)
    t_idx[i] = (round(tT[0], 3) - start_t) * step_t
    T_i = (round(tT[1], 3) - start_T) * step_T
    Ruv_tT[i] = R_uv[int(t_idx[i]),int(T_i)]

    # step_t = R2_w.shape[0]/(end_t-start_t)
    # step_T = R2_w.shape[1]/(end_T-start_T)
    # t_i = (round(tT[0], 1) - start_t) * step_t 
    # T_i = (round(tT[1], 1) - start_T) * step_T 
    # Rw_tT[i] = R2_w[int(t_i), int(T_i)]

print(tT_uv)
print(Ruv_tT)
# print(Rw_tT)

with open(f'{path}orbs.txt', 'a') as file:
    for i, tT in enumerate(tT_uv):
        T = round(tT[1], 3)
        t = round(tT[0], 3)
        file.write(f'{i},{T},{t},{int(t_idx[i])},{round(Ruv_tT[i],3)}\n') #,{round(Rw_tT[i],3)

# with open(f'{path}orbs_3.txt', 'x') as file:
#     file.write(f'i, T, t, R_w\n')


# # plot_R(R2_w, 'R_\omega', start_t, end_t, start_T, end_T)
# # plt.show()

# start_t, end_t, start_T, end_T = 2000,3900, .05, 50
# plot_R(R3_w, 'R_\omega', start_t, end_t, start_T, end_T)

# #Save local minimums as guesses for newton solver 
# tT_w = plt.ginput(n=0, timeout=0, mouse_add=None, mouse_pop=None, mouse_stop=None)
# tT_w = np.array(tT_w)
# tT_w = tT_w[tT_w[:,1].argsort()]

# #Save R values for the guesses
# Rw3_tT = np.zeros(len(tT_w))

# for i, tT in enumerate(tT_w):
#     step_t = R3_w.shape[0]/(end_t-start_t)
#     step_T = R3_w.shape[1]/(end_T-start_T)
#     t_i = (round(tT[0], 1) - start_t) * step_t 
#     T_i = (round(tT[1], 1) - start_T) * step_T 
#     Rw3_tT[i] = R3_w[int(t_i), int(T_i)]

# print(tT_w)
# print(Rw3_tT)

# with open(f'{path}orbs_3.txt', 'a') as file:
#     for i, tT in enumerate(tT_w):
#         T = round(tT[1], 1)
#         t = round(tT[0], 1)
#         file.write(f'{i},{T},{t},{Rw3_tT[i]}\n')


# with open(f'{path}orbs.txt', 'x') as file:
#     file.write(f'i, T, t, R_w\n')


# with open(f'{path}orbs.txt', 'a') as file:
#     for i, tT in enumerate(tT_w):
#         T = round(tT[1], 1)
#         t = round(tT[0], 1)
#         file.write(f'{i},{T},{t},{Rw3_tT[i]}\n')




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
