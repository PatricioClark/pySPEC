
import numpy as np
import matplotlib.pyplot as plt

T = 499
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

print('Saved fields')
np.savez(f'output/fields_t.npz', uu_t=uu_t,vv_t=vv_t)