import numpy as np

f_name = 'fields_0'
fields = np.load(f'{f_name}.npz')
uu = fields['uu'].T
vv = fields['vv'].T

idx = 0
ext = 7
np.save(f'uu_{idx:0{ext}}', uu)
np.save(f'vv_{idx:0{ext}}', vv)
