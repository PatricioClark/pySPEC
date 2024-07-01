
import numpy as np
import matplotlib.pyplot as plt

#for output2
T = 1499
stat = 500_000
save_freq = 50
#for output3:
# T = 1990
# stat = 2_000_000
# save_freq = 100

dt = 1e-3
nx = ny = 256
Nt = int(T/dt //save_freq)

uu_t = np.zeros((Nt,nx,ny))
vv_t = np.zeros((Nt,nx,ny))

path = '/share/data4/jcullen/pySPEC/run1/'
for i in range(Nt):
    step = stat + save_freq * i
    f_name = f'fields_{step:07}'
    fields = np.load(f'{path}output2/{f_name}.npz')
    uu = fields['uu']
    vv = fields['vv']
    uu_t[i,:,:] = uu
    vv_t[i,:,:] = vv

print('Saved fields')
np.savez(f'output/fields_t.npz', uu_t=uu_t,vv_t=vv_t)