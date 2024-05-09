import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from gmres_kol import GMRES

### Feed initial aproximation extracted with Kolmogorov solver
path = '/share/data4/jcullen/pySPEC/run1/'
t_i = 135
dt = 1e-3
f_name = f'fields_{int(t_i/dt):06}'
fields = np.load(f'{path}output/{f_name}.npz')
uu = fields['uu']
vv = fields['vv']
X = np.concatenate((uu.flatten(), vv.flatten()))


