import numpy as np
import matplotlib.pyplot as plt
import params2D as pm
import mod2D as mod
import os
import subprocess

# Initialize solver
grid = mod.Grid(pm)

def get_var_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable][0] 

def vort(uu, vv):
    fu, fv = mod.forward(uu), mod.forward(vv)
    uy = mod.inverse(mod.deriv(fu, grid.ky))
    vx = mod.inverse(mod.deriv(fv, grid.kx))
    return uy - vx

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

def mkplotdirs(dir_name, nstep = False):
    '''make plotting directories'''
    if not nstep:
        mkdir(f'{dir_name}')
        mkdir(f'{dir_name}/newt_gmres')
    else:
        path = f'{dir_name}/nstep{nstep}'
        for folder in ('', '/ti_tf', '/evol', '/Ek', '/balance'):
            mkdir(path+folder)

def saveplot(field, title, path):
    '''save plot of field'''
    plt.figure(figsize = (13,10))
    plt.imshow(field, origin = 'lower', extent = [0, pm.Nx, 0, pm.Ny])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(title)
    plt.savefig(f'{path}/{title}.png')
    plt.close()

def plot_ti_tf(nstep, dir_name):
    '''plot fields at time t (ti) and time t+T (tf) for newton step nstep'''
    fields = np.load(f'../{dir_name}/output/fields_{nstep}.npz')
    uu = fields['uu']
    vv = fields['vv']
    oz = vort(uu, vv)

    fields = np.load(f'../{dir_name}/output/fields_{nstep}_T.npz')
    uu2 = fields['uu']
    vv2 = fields['vv']
    oz2 = vort(uu2, vv2)
    # print('Newton step = ', nstep)
    # print('|u(t)|=', np.linalg.norm(uu))
    # print('|u(t+T)|=', np.linalg.norm(uu2))
    # print('|u(t+T)-u(t)|=', np.linalg.norm(uu-uu2))

    path = f'{dir_name}/nstep{nstep}'

    saveplot(uu, 'u(t)', path+'/ti_tf')
    saveplot(vv, 'v(t)', path+'/ti_tf')
    saveplot(oz, 'w(t)', path+'/ti_tf')

    saveplot(uu2, 'u(t+T)', path+'/ti_tf')
    saveplot(vv2, 'v(t+T)', path+'/ti_tf')
    saveplot(oz2, 'w(t+T)', path+'/ti_tf')

    saveplot(uu2-uu, 'u(t+T)-u(t)', path+'/ti_tf')
    saveplot(vv2-vv, 'v(t+T)-v(t)', path+'/ti_tf')
    saveplot(oz2-oz, 'w(t+T)-w(t)', path+'/ti_tf')

def get_T(nstep, dir_name):
    '''get period T from newton step nstep'''
    solver = np.loadtxt(f'../{dir_name}/prints/solver.txt', delimiter = ',', skiprows = 1)
    return solver[nstep,1]

def GetSpacedElements(list, numElems = 4):
    '''returns evenly spaced elements from list'''
    idxs = np.round(np.linspace(0, len(list)-1, numElems)).astype(int)
    return [list[idx] for idx in idxs] 

def plot_evol(nstep, dir_name):
    '''plots a few frames of the orbit evolution'''
    plot_path = f'{dir_name}/nstep{nstep}/evol/' 
    evol_path = f'../{dir_name}/evol/nstep{nstep}'
    listdir = os.listdir(evol_path)
    listdir.sort()

    # #me quedo con 5 plots equiespaciados temporalmente
    listdir = GetSpacedElements(listdir, 5)

    for file in listdir:
        full_path = os.path.join(evol_path, file)
        oz = np.load(full_path)        
        step = file.split('.')[0][3:]#get the step number from file name
        t = round(int(step)*pm.dt,2)
        saveplot(oz, f'oz.t_{t}',plot_path)

def plot_balance(nstep, dir_name):
    '''plots energy, enstrophy, injection rate from a newton step nstep'''
    time, eng, ens, inj =  np.loadtxt(f'../{dir_name}/balance{nstep}.dat', skiprows=1,delimiter = ',', unpack = True)
    path = f'{dir_name}/nstep{nstep}/balance'
    plt.figure(figsize = (8,5))
    plt.plot(time, eng)
    plt.xlabel('Time')
    plt.ylabel('Energy')

    plt.savefig(path+'/energy.png')
    plt.close()

    plt.figure(figsize = (8,5))
    plt.plot(time, inj)
    plt.xlabel('Time')
    plt.ylabel('Injection')
    plt.savefig(path+'/injection.png')
    plt.close()

    plt.figure(figsize = (8,5))
    plt.plot(time, ens)
    plt.xlabel('Time')
    plt.ylabel('Enstrophy')
    plt.savefig(path+'/enstrophy.png')
    plt.close()

def plot_mean_spectrum(nstep, dir_name):
    '''plots mean spectrum of an orbit in a newton step nstep'''
    path = f'../{dir_name}/Ek/nstep{nstep}'
    fnames = os.listdir(path)
    Eks = np.zeros((len(fnames), pm.Nx//2))
    for i, file in enumerate(fnames):
        full_path = os.path.join(path, file)
        Eks[i,:] = np.load(full_path)
    Ek = np.mean(Eks, axis = 0)

    path = f'../{dir_name}/nstep{nstep}/Ek'    
    plt.figure(figsize = (8,5))
    plt.loglog(Ek)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$<E(k)>_t$')
    plt.savefig(path+'/mean_spectrum.png')
    plt.close()


def plot_newt_gmres(dir_name):
    '''plot convergence data, gmres error, newton error, hookstep, orbit quantities'''
    plot_path = f'{dir_name}/newt_gmres/'

    b_error = np.loadtxt(f'../{dir_name}/prints/b_error.txt', delimiter = ',', skiprows = 1)
    solver = np.loadtxt(f'../{dir_name}/prints/solver.txt', delimiter = ',', skiprows = 1)

    iter_newt = b_error[:,0]

    #plot |b| vs newt_it
    plt.figure(figsize=(8,5))
    plt.scatter(iter_newt, b_error[:,1])
    plt.xlabel('Iter. Newton')
    plt.ylabel(r'$|\delta \! X|$')#|b|
    plt.savefig(plot_path+'b_error.png')
    plt.close()

    plt.figure(figsize=(8,5))
    plt.scatter(iter_newt, b_error[:,1])
    plt.yscale('log')
    plt.grid('on')
    plt.xlabel('Iter. Newton')
    plt.ylabel(r'$|\delta \! X|$')#|b|
    plt.savefig(plot_path+'b_error_log.png')
    plt.close()

    #plot T vs newt_it
    plt.figure(figsize=(8,5))
    plt.scatter(iter_newt, solver[:,1])
    plt.xlabel('Iter. Newton')
    plt.ylabel(r'$T$')
    plt.savefig(plot_path+'T.png')
    plt.close()

    #plot sx vs newt_it
    plt.figure(figsize=(8,5))
    plt.scatter(iter_newt, solver[:,2])
    plt.xlabel('Iter. Newton')
    plt.ylabel(r'$s_x$')
    plt.savefig(plot_path+'sx.png')
    plt.close()

    #plot gmres data
    plt.figure(figsize=(8,5))
    list_iters = [i for i in range(1, int(max(iter_newt)+1))]
    spaced_iters = GetSpacedElements(list_iters, 4)
    iter_max = np.zeros(len(list_iters))
    for i, iter in enumerate(list_iters):
        gmres_iter, error = np.loadtxt(f'../{dir_name}/prints/error_gmres/iter{iter}.txt', delimiter = ',', skiprows = 1, ndmin = 2, unpack = True)
        iter_max[i] = np.max(gmres_iter)
        if iter in spaced_iters:
            plt.scatter(gmres_iter,error, label = f'Iter. {iter}')
    plt.xlabel('Iter. GMRes')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')    
    plt.savefig(plot_path+'error_gmres.png')
    plt.close()

    #plot iter max gmres    
    plt.figure(figsize=(8,5))
    plt.bar(list_iters, iter_max)
    plt.xlabel('Iter. Newton')
    plt.ylabel('Cantidad iter. GMRes')
    plt.savefig(plot_path+'max_iter_gmres.png')
    plt.close()
    
    #plot hookstep # iterations vs newton iter
    iter_max_hook = np.zeros(len(list_iters))
    for i, iter in enumerate(list_iters):
        iter_hook, _ = np.loadtxt(f'../{dir_name}/prints/hookstep/iter{iter}.txt', delimiter = ',', skiprows = 1, ndmin = 2, unpack = True)
        iter_max_hook[i] = np.max(iter_hook)
    plt.figure(figsize=(8,5))
    plt.bar(list_iters, iter_max_hook)
    plt.xlabel('Iter. Newton')
    plt.ylabel('Cantidad iter. Hookstep')
    plt.savefig(plot_path+'max_iter_hook.png')
    plt.close()

    #plot hookstep for first and last newton step    
    iters = (list_iters[0], list_iters[len(list_iters)//2],list_iters[-1])
    colors = ('r', 'b', 'g')
    plt.figure(figsize=(8,5))
    for it, color in zip(iters, colors):
        #load hook for max iter (indexing 1)
        iter_hook, dX = np.loadtxt(f'../{dir_name}/prints/hookstep/iter{it}.txt', delimiter = ',', skiprows = 1, ndmin = 2, unpack = True)
        #load previous |dX| (that hookstep has to improve)
        _,b_error = np.loadtxt(f'../{dir_name}/prints/b_error.txt', delimiter = ',', skiprows = 1, unpack = True)

        plt.scatter(iter_hook, dX, color = color, label = fr'$|\delta \! X_{{{it}}} hook|$')
        plt.grid('on')
        plt.xlabel('Iter. Hookstep')
        plt.ylabel(r'$|\delta \! X_{hook}|$')
        #it-1 to compare with previous newton step
        plt.hlines(b_error[it-1], iter_hook[0], iter_hook[-1],linestyles= 'dashed', color= color,label = fr'$|\delta \! X_{it-1}|$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(plot_path+'hookstep.png')
    plt.close()

    
    # labels = ('|dX|','|dY_dX|','|dX_dt|','|dY_dT|','t_proj')
    # for i in range(1,int(max(iter_newt)+1)):
    #     apply_A = np.loadtxt(f'../{dir_name}/prints/apply_A/iter{i}.txt', delimiter = ',', skiprows = 1)
    #     for j in range(5):
    #         plt.figure(j, figsize=(8,5))
    #         plt.scatter(apply_A[:,0],apply_A[:,1], label = f'Iter. {i}')
    #         plt.xlabel('Iter. GMRes')
    #         plt.ylabel(f'{labels[j]}')
    #         plt.legend()
    #         plt.savefig(plot_path+'apply_A/{labels[j]}.png')



#Get info from txt with convergence info
orbs, bs_min, iters_min, _ = np.loadtxt('../data_orbs.txt', delimiter = ',', skiprows=1, unpack = True)

for orb_n, b_min, iter_min in zip(orbs, bs_min, iters_min):
    #Classify by convergence of orbits
    # if b_min < 0.0085:

    #Get info from txt with initial orbits info
    data = np.loadtxt('../ginput_data_orbs.txt', delimiter=',', skiprows=1, comments='-', usecols = (0,1,2))
    orb = data[int(orb_n)] #selecciona fila con info de orb_n
    dir_name = f'orb{int(orb[0]):02}_{orb[1]}' #directory with name orb09_5.9 where 09 is the orb nmb and 5.9 the guessed T


    #make plots for convergence data
    # mkplotdirs(dir_name)
    # plot_newt_gmres(dir_name)

    #make plots for initial state and converged state
    # for nstep in (0, int(iter_min)):
        # mkplotdirs(dir_name,nstep)
        # plot_balance(nstep,dir_name)
        # plot_mean_spectrum(nstep,dir_name)
        # plot_ti_tf(nstep,dir_name)
        # plot_evol(nstep,dir_name)
