import numpy as np
import matplotlib.pyplot as plt

def get_var_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable][0] 


step = 1000
f_name = f'fields_{step:06}'
fields = np.load(f'output/{f_name}.npz')

uu = fields['uu']
vv = fields['vv']

print(uu.shape)

rmp = 256**2
fu_r = fields['fu'].real/rmp
fv_r = fields['fv'].real/rmp
fu_i = fields['fu'].imag/rmp
fv_i = fields['fv'].imag/rmp

for f in (uu, vv, fu_r, fv_r, fu_i, fv_i):
    plt.figure()
    plt.imshow(f)
    plt.title(f'{get_var_name(f)}_step{step}')
    plt.colorbar()
plt.show()

