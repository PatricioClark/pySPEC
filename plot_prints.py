import numpy as np
import matplotlib.pyplot as plt

def get_var_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable][0] 

# with open('prints/b_error.txt', 'r') as file1, open('prints/solver.txt', 'r') as file2, \
#         open('prints/error_gmres.txt', 'r') as file4, open('prints/hookstep.txt', 'a') as file4,\
#         open('prints/apply_A.txt', 'a') as file5:

b_error = np.loadtxt('prints/b_error.txt', delimiter = ',', skiprows = 1)
solver = np.loadtxt('prints/solver.txt', delimiter = ',', skiprows = 1)

iter_newt = b_error[:,0]

plt.figure()
plt.plot(iter_newt, b_error[:,1])
plt.xlabel('Iter. Newton')
plt.ylabel('|b|')
plt.savefig(f'plots/prints_plots/b_error.png')
plt.close()

plt.figure()
plt.plot(iter_newt, solver[:,1])
plt.xlabel('Iter. Newton')
plt.ylabel('T')
plt.savefig(f'plots/prints_plots/T.png')

plt.figure()
plt.plot(iter_newt, solver[:,2])
plt.xlabel('Iter. Newton')
plt.ylabel('|X|')
plt.savefig(f'plots/prints_plots/norm_X.png')

plt.figure()
for i in range(1,len(iter_newt)):
    err_gmres = np.loadtxt(f'prints/error_gmres/iter{i}.txt', delimiter = ',', skiprows = 1)
    plt.plot(err_gmres[:,0],err_gmres[:,1], label = f'Iter. {i}')

plt.xlabel('Iter. GMRes')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig(f'plots/prints_plots/error_gmres.png')

plt.figure()
for i in range(1,len(iter_newt)):
    hookstep = np.loadtxt(f'prints/hookstep/iter{i}.txt', delimiter = ',', skiprows = 1)
    plt.plot(hookstep[:,0],hookstep[:,1], label = f'Iter. {i}')

plt.xlabel('Iter. Hookstep')
plt.ylabel('|b|')
plt.legend()
plt.savefig(f'plots/prints_plots/hookstep.png')

labels = ('|dX|','|dY/dX|','|dX/dt|','|dY/dT|','t_proj')
for i in range(1,len(iter_newt)):
    apply_A = np.loadtxt(f'prints/apply_A/iter{i}.txt', delimiter = ',', skiprows = 1)
    for j in range(5):
        plt.figure(j)
        plt.plot(apply_A[:,0],apply_A[:,1], label = f'Iter. {i}')
        plt.xlabel('Iter. GMRes')
        plt.ylabel(f'{labels[j]}')
        plt.legend()
        plt.savefig(f'plots/prints_plots/apply_A/{labels[j]}.png')