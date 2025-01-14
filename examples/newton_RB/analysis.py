import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys
import os
import json
from types import SimpleNamespace

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx


def plot_diagnostic(iN):
    path = f'balance/iN{iN:02}'
    t, div, vt1, vt2, vn1, vn2 = np.loadtxt(f'{path}/noslip_diagnostic.txt',unpack=True)
    _, th1, th2  = np.loadtxt(f'{path}/scalar_constant_diagnostic.txt',unpack=True)
    t[0]= t[1] = t[2] - pm.dt
    t[-1] = t[-2] + pm.dt

    plt.figure()
    plt.plot(t, div, '-o', label = 'div(v)')
    plt.plot(t, vt1, '-o', label = r'$v_t|_{z=0}$')
    plt.plot(t, vt2, '-o', label = r'$v_t|_{z=L}$')
    plt.plot(t, vn1, '-o', label = r'$v_n|_{z=0}$')
    plt.plot(t, vn2, '-o', label = r'$v_n|_{z=L}$')
    plt.plot(t, th1, '-o', label = r'$th|_{z=0}$')
    plt.plot(t, th2, '-o', label = r'$th|_{z=L}$')
    plt.xlabel('$t$')
    plt.yscale('log')
    plt.legend(loc = 'lower right')

    os.makedirs(f'Imgs/diag', exist_ok = True)
    plt.savefig(f'Imgs/diag/diag_{iN:02}.png', dpi = 300)
    plt.close()

def plot_point(iN, position):
    '''plots time series and Fourier transform of a point in the domain
    comparing iN=0 and iN'''	

    types = ["th", "vx", "vz"]
    shape = (pm.Nx, pm.Ny, pm.Nz)
    x,z = position

    def def_vars(iN):
        #iN last. get # files
        files = os.listdir(f"output/iN{iN:02}/")
        n_files = len(files)//(len(types)+2) #because of pr and vy    
        t = np.arange(n_files)*pm.ostep*pm.dt 
        w = np.fft.rfftfreq(len(t), pm.dt)
        point = np.zeros((len(types), n_files)) 
        f_data = np.zeros((len(types), len(w)))
        return n_files, t, w, point, f_data

    def unpack_point(iN, type_data, nmb, shape):
        file_ = f'output/iN{iN:02}/' + '{}.{:05d}.out'.format(type_data, nmb)
        data = np.fromfile(file_, dtype=np.float64).reshape(shape,order='F')
        return data[x,0,z]

    n_files, t, w, point, f_data = def_vars(iN)
    n_files0, t0, w0, point0, f_data0 = def_vars(0)

    #get data of point of interest (x, z coordinates)
    fig, axs = plt.subplots(len(types), 2, figsize = (12,6))
    ax1, ax2 = axs[:,0], axs[:,1]
    ms = 3 # marker size

    for i, type_data in enumerate(types):
        for nmb in range(n_files): 
            point[i, nmb] = unpack_point(iN, type_data, nmb, shape)  
        for nmb in range(n_files0): 
            point0[i, nmb] = unpack_point(0, type_data, nmb, shape)  
    
        f_data[i,:] = np.fft.rfft(point[i,:])
        f_data0[i,:] = np.fft.rfft(point0[i,:])

        Delta = np.abs(point[i,-1] - point[i,0])
        Delta0 = np.abs(point0[i,-1] - point0[i,0])
        ax1[i].plot(t0, point0[i,:], marker = 'o', ms = ms, label = fr'$i_N=0. \Delta={Delta0:.1e}$')
        ax1[i].plot(t, point[i,:], marker = 'o', ms = ms, label = fr'$i_N={iN}. \Delta={Delta:.1e}$')
        ax1[i].set_ylabel(rf'${type_data}$')
        ax1[i].set_xlabel(r'$t$')
        # ax1[i].set_title(fr'$|{type_data}_f- {type_data}_i|={point$')
        ax1[i].legend(fontsize = 8)

        f_data0[i,:] = np.fft.rfft(point0[i,:])
        ax2[i].loglog(w0, np.abs(f_data0[i,:])**2, marker = 'o', ms = ms, label = fr'$i_N=0$')
        ax2[i].loglog(w, np.abs(f_data[i,:])**2, marker = 'o', ms = ms, label = fr'$i_N={iN:02}$')
        ax2[i].set_ylabel(fr'$ \mathcal{{F}}({type_data})$')
        ax2[i].set_xlabel(r'$\omega$')
        ax2[i].legend()

    plt.suptitle(f'Point ({x},0,{z})')
    plt.tight_layout()
    path = 'Imgs/balance'
    os.makedirs(path, exist_ok = True)
    plt.savefig(f'{path}/point_{iN:02}.png', dpi = 300)
    plt.close()

def plot_fields(iN, n_fields):
    types = ["th", "vx", "vz"]
    extent = [0, pm.Lx, 0, pm.Lz]

    # nmbs = list(range(0, n_fields, step)).append(n_fields)
    nmbs = [n_fields]
    shape = (pm.Nx, pm.Ny, pm.Nz)

    for nmb in nmbs:
        for i, type_data in enumerate(types):

            file_ = f"output/iN{iN:02}/{type_data}.{nmb:05}.out"
            data = np.fromfile(file_, dtype=np.float64).reshape(shape,order='F')
            
            plt.figure()
            plt.imshow(data[:,0,:].T, extent=extent, vmin = -2., vmax = 2.)
            plt.title(f'{type_data} en step={nmb}')
            plt.colorbar()

            os.makedirs(f'Imgs/output/iN{iN:02}', exist_ok = True)
            plt.savefig(f'Imgs/output/iN{iN:02}/{type_data}.{nmb:05}.png', dpi = 300)
            plt.close() 

def plot_vec_vel(nmb, h, save=False, plot=True):
    '''plots velocity field at time step nmb'''
    shape = (pm.Nx, pm.Ny, pm.Nz)
    fig, ax = plt.subplots()

    types = ["vx", "vz"]
    extent = [0, 2*np.pi, 0, 3.14]

    ax.set_xlabel("x")
    ax.set_ylabel("z")

    file_ = "bin/output/" + '{}.{:05d}.out'.format("vx", nmb)
    vx = np.fromfile(file_, dtype=np.float64).reshape(shape,order='F')
    file_ = "bin/output/" + '{}.{:05d}.out'.format("vz", nmb)
    vz = np.fromfile(file_, dtype=np.float64).reshape(shape,order='F')

    # Define a stride to reduce the density of arrows
    stride = 10

    # Generate a grid of coordinates based on the shape of vx, using the stride
    y, x = np.meshgrid(np.arange(0, vx.shape[1], stride), np.arange(0, vx.shape[0], stride))

    # Slice the velocity matrices according to the stride as well
    vx_sliced = vx[::stride, ::stride]
    vz_sliced = vz[::stride, ::stride]

    vx_sliced = vx_sliced.squeeze()
    vz_sliced = vz_sliced.squeeze()
    print(vx_sliced.shape)
    print(vz_sliced.shape)

    # The quiver function will plot arrows; adjust the scale to make them visible.
    widths = np.linspace(0, 200, x.size)
    ax.quiver(x, y, vx_sliced, vz_sliced, linewidth=widths)

    # Set plot limits
    plt.xlim(-1, vx.shape[1])
    plt.ylim(-1, vx.shape[0])

    # Labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Field')

    plt.show()

def plot_balance(iN):
    t, v2, w2, eps = np.loadtxt(f'balance/iN{iN:02}/balance.txt',unpack=True)
    t[0]= t[1] = t[2] - pm.dt
    t[-1] = t[-2] + pm.dt
    eng = v2*.5
    dis = w2

    fig,axes = plt.subplots(2,1)
    axes = axes.flatten()

    axes[0].plot(t, eng)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$E$")
    axes[0].set_title(f"$|E_f - E_i|={abs(eng[-1]-eng[0]):.2e}$")

    # Enstrofia
    axes[1].plot(t, dis)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("$D$")
    axes[1].set_title(f"$|D_f - D_i|={abs(dis[-1]-dis[0]):.2e}$")

    plt.tight_layout()

    os.makedirs(f'Imgs/balance', exist_ok = True)
    plt.savefig(f'Imgs/balance/balance_{iN:02}.png', dpi = 300)
    plt.close()

def balance_vs_init(iN):
    '''plots E, D of pre vs post convergence, and energy time spectrum'''
    fig = plt.figure(figsize = (12,6))#, layout = 'constrained')

    gs = GridSpec(2, 2, figure=fig, wspace = .3)#, width_ratios=)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1,1])

    font_size = 14

    #load eng, ens at nstep=0, nstep=iter_min
    for i_n in (0, iN):
        t, v2, w2, eps = np.loadtxt(f'balance/iN{i_n:02}/balance.txt',unpack=True)
        t[0]= t[1] = t[2] - pm.dt
        t[-1] = t[-2] + pm.dt
        eng = v2*.5 #Remove first because its duplicated, last because t is miscalculated because of dt modification
        dis = w2 #dissipation

        Delta_E, Delta_D = np.abs(eng[-1]-eng[0]), np.abs(dis[-1]-dis[0])
        ax1.plot(t, eng, 'o-', label = fr'$i_N={i_n}. \Delta={Delta_E:.1e}$')
        ax2.plot(t, dis, 'o-', label = fr'$i_N={i_n}. \Delta={Delta_D:.1e}$')
        # ax1.plot(t, eng-np.mean(eng), 'o-', label = fr'$i_N={iN}. \Delta={Delta_E:.1e}$')
        # ax2.plot(t, dis-np.mean(dis), 'o-', label = fr'$i_N={iN}. \Delta={Delta_D:.1e}$')

        #calculate Fourier of energy in time
        f_eng = np.fft.rfft(eng)
        f_eng = np.abs(f_eng)**2
        w = np.fft.rfftfreq(len(eng), pm.dt)
        ax3.loglog(w[1:], f_eng[1:], 'o-', label = fr'$i_N={i_n}$')    

        #calculate Fourier of dissipation in time
        f_dis = np.fft.rfft(dis)
        f_dis = np.abs(f_dis)**2
        w = np.fft.rfftfreq(len(dis), pm.dt)
        ax4.loglog(w[1:], f_dis[1:], 'o-', label = fr'$i_N={i_n}$')    

    ax1.set_ylabel(r'$E$', fontsize = font_size)
    # ax1.set_ylabel(r'$E- \langle E \rangle_t$', fontsize = font_size)

    ax2.set_xlabel(r'$t$', fontsize = font_size)
    ax2.set_ylabel(r'$D$', fontsize = font_size)
    # ax2.set_ylabel(r'$D-\langle D \rangle_t$', fontsize = font_size)

    ax3.set_ylabel(r'$\mathcal{F}(E)$', fontsize = font_size)

    ax4.set_xlabel(r'$\omega$', fontsize = font_size)
    ax4.set_ylabel(r'$\mathcal{F}(D)$', fontsize = font_size)

    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=font_size)
        ax.legend(fontsize = 10)

    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = font_size*.85)

    plt.tight_layout()
    plt.savefig(f'Imgs/balance/balance_{iN}_vs_0.png', dpi = 300)
    # plt.show()
    plt.close()



def calc_max_k(x, z):
    shape = (pm.Nx, pm.Ny, pm.Nz)

    delta_x = x / shape[0]
    delta_z = z / shape[2]

    kx_max = 2*np.pi / delta_x / 3
    kz_max = 2*np.pi / delta_z / 3

    return kx_max, kz_max


def calc_dif_L2(path1, path2, nmb1, nmb2):
    shape = (pm.Nx, pm.Ny, pm.Nz)

    types = ["th", "vx", "vz"]
    extent = [0, 2*np.pi, 0, 3.14]
    diff = []

    for i, type_data in enumerate(types):
        i += 1
        file1 = path1 + '{}.{:05d}.out'.format(type_data, nmb1)
        data1 = np.fromfile(file1, dtype=np.float64).reshape(shape,order='F')
        file2 = path2 + '{}.{:05d}.out'.format(type_data, nmb2)
        data2 = np.fromfile(file2, dtype=np.float64).reshape(shape,order='F')

        diff.append(np.linalg.norm(data2-data1))
    return np.array(diff)

def tseries_dif_L2():
    '''plots time series of difference between runs in L2 norm'''
    t_idx = 1000
    idx_range = 82
    shape = (pm.Nx, pm.Ny, pm.Nz)

    
    diff = np.empty((idx_range, 3))
    for i in range(idx_range):
        nmb1 = i
        nmb2 = i + t_idx 
        path1 = 'output/iN00/'
        path2 = '/share/data7/mvinograd/SPECTER/512_ra_1e7/bin/output/'
        
        diff[i,:] = calc_dif_L2(shape, path1, path2, nmb1, nmb2)

    plt.figure()
    for i, type_data in enumerate(['th', 'vx', 'vz']):
        plt.plot(diff[:,i], label = type_data)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('t_idx')
    plt.ylabel('L2 dif')
    plt.savefig('Imgs/L2_dif.png')
    plt.show()

def dif_X_L2(nmb1, nmb2):
    shape = (pm.Nx, pm.Ny, pm.Nz)

    types = ["th", "vx", "vz"]
    extent = [0, 2*np.pi, 0, 3.14]

    for i, type_data in enumerate(types):
        i += 1
        file1 = "bin/output/" + '{}.{:05d}.out'.format(type_data, nmb1)
        data1 = np.fromfile(file1, dtype=np.float64).reshape(shape,order='F')
        
        file2 = "bin/output/" + '{}.{:05d}.out'.format(type_data, nmb2)
        data2 = np.fromfile(file2, dtype=np.float64).reshape(shape,order='F')
        
        if i == 1:
            X1 = data1[np.newaxis,...]
            X2 = data2[np.newaxis,...]
        else:
            X1 = np.concatenate((X1, data1[np.newaxis,...]), axis = 0)
            X2 = np.concatenate((X2, data2[np.newaxis,...]), axis = 0)

        plt.figure()
        plt.imshow(data1[:,0,:].T, extent=extent, vmin = -1.7, vmax = 1.7)
        plt.title(f'{type_data} en step={nmb1}. ||{type_data}||={np.round(np.linalg.norm(data1),2)}')
        plt.colorbar()
        plt.savefig(f'Imgs/dif_t1_t2/{type_data}.{nmb1}.png')
        plt.close()

        plt.figure()
        plt.imshow(data2[:,0,:].T, extent=extent, vmin = -1.7, vmax = 1.7)
        plt.title(f'{type_data} en step={nmb2}. ||{type_data}||={np.round(np.linalg.norm(data2),2)}')
        plt.colorbar()
        plt.savefig(f'Imgs/dif_t1_t2/{type_data}.{nmb2}.png')
        plt.close()

        plt.figure()
        plt.imshow(data2[:,0,:].T- data1[:,0,:].T, extent=extent)
        plt.title(f'{type_data} en step={nmb2}.||{type_data} dif||={np.round(np.linalg.norm(data2-data1))}')
        plt.colorbar()
        plt.savefig(f'Imgs/dif_t1_t2/{type_data}_dif.{nmb2}_{nmb1}.png')
        plt.close()

    dif_X = X2 - X1
    print(np.linalg.norm(dif_X))

def plot_error():
    iter_newt, b_error = np.loadtxt(f'prints/b_error.txt', delimiter = ',', skiprows = 1, unpack = True)
    
    plt.figure()
    plt.semilogy(iter_newt, b_error, marker = 'o')
    plt.xlabel(r'$i_N$')
    plt.ylabel(r'$\Vert \mathbf{F} \Vert$')
    plt.savefig('Imgs/b_error.png',  dpi = 300)
    plt.close()

def get_vmin_vmax(n_newt):
    vmins = []
    vmaxs = []
    types = ['vx','vz','th']
    for type_ in types:
        # First pass: Determine global vmin and vmax
        vmin = float('inf')
        vmax = float('-inf')
        for iN in range(n_newt):
            fields0 = np.load(f'output/fields_{iN:02}.npz')            
            fieldsT = np.load(f'output/fields_{iN:02}_T.npz')            

            data0 = fields0[type_]
            dataT = fieldsT[type_]
             
            Delta = (dataT - data0)[:,0,:]
            vmin = min(vmin, np.min(Delta))
            vmax = max(vmax, np.max(Delta))

        vmins.append(vmin)
        vmaxs.append(vmax)
    return vmins, vmaxs

def plot_Delta(iN, vmins, vmaxs):
    #save Delta file
    types = ['vx','vz','th']
    extent = [ 0, 2*np.pi, 0, 3.1419]
    path  = f'Imgs/Delta/'
    os.makedirs(path, exist_ok = True)
    for i, type_ in enumerate(types):
        fields0 = np.load(f'output/fields_{iN:02}.npz')            
        fieldsT = np.load(f'output/fields_{iN:02}_T.npz')            

        data0 = fields0[type_]
        dataT = fieldsT[type_]
         
        Delta = np.abs((dataT - data0)[:,0,:])
        
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), constrained_layout = True)
        ax = axs[0]
        cax = ax.imshow(Delta.T, origin = 'lower', extent = extent, vmin = 0., vmax = max(vmaxs[i], -vmins[i]))
        plt.colorbar(cax, ax = ax)
        ax = axs[1]
        cax = ax.imshow(Delta.T, origin = 'lower', extent = extent)
        plt.colorbar(cax, ax = ax)
        
        plt.suptitle(f'mean|{type_}| = {np.mean(np.abs(data0)):.3f}')
        plt.savefig(path+f'{type_}_{iN:02}.png', dpi = 300)
        plt.close()

def GetSpacedElements(list, numElems = 4):
    '''returns evenly spaced elements from list'''
    idxs = np.round(np.linspace(0, len(list)-1, numElems)).astype(int)
    return [list[idx] for idx in idxs] 


def plot_errors(n_newt = None):
    '''plots newton errors and gmres errors'''
    
    fig, axs = plt.subplots(1, 2, figsize=(15,6.5))#, constrained_layout = True)

    font_size = 18

    #plot |b|:
    iter_newt, b_error = np.loadtxt(f'prints/solver.txt', delimiter = ',', skiprows = 1, unpack = True, usecols = (0,1))
    
    #remove repeated (restart) values
    _,idxs = np.unique(iter_newt, return_index=True)
    iter_newt, b_error = iter_newt[idxs], b_error[idxs]

    if n_newt:
        iter_newt = iter_newt[:n_newt]
        b_error = b_error[:n_newt]

    #plot |b| vs newt_it
    ax = axs[0]

    # cmap = plt.get_cmap('viridis')
    alpha = 1
    ax.plot(iter_newt, b_error, color = 'royalblue', alpha = alpha, marker = 'o')

    ax.set_yscale('log')
    ax.set_xlabel(r'$i_N$', fontsize = font_size)
    ax.set_ylabel(r'$\Vert \mathbf{F} \Vert$', fontsize = font_size, labelpad = 8)#TODO: aclaro de alguna forma que es el relativo?

    # Set font size for ticks
    ax.tick_params(axis='both', labelsize=font_size)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #plot gmres data
    ax = axs[1]
    # list_iters = [i for i in range(1, int(max(iter_newt)+1))]
    list_iters = [i for i in range(1, int(max(iter_newt)))]
    # list_iters = GetSpacedElements(list_iters, 6)

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    max_iter = 0
    for i, iter in enumerate(list_iters):
        gmres_iter, error = np.loadtxt(f'prints/error_gmres/iN{iter:02}.txt',\
                            delimiter = ',', skiprows = 1, ndmin = 2, unpack = True)
        # if iter in spaced_iters:
        max_iter = max(max_iter, gmres_iter[-1])
        ax.plot(gmres_iter,error, marker = 'o', color = cmap(i/len(list_iters)),label = fr'$i_N = {iter}$')

    # ax.hlines(10**(-5), gmres_iter[0], max_iter+1, ls = 'dashed', color = 'k')
    ax.set_xlabel(r'$i_G$', fontsize = font_size)
    ax.set_ylabel(r'$\Vert A\delta \mathbf{x} - \mathbf{b} \Vert / \, \Vert \mathbf{b} \Vert$', fontsize = font_size, labelpad = 8)
    # ax.set_ylabel(r'$\Vert A_{i_N}x - b_{i_N} \Vert / \, \Vert b_{i_N} \Vert$', fontsize = font_size)
    # ax.set_ylabel(r'$\frac{\Vert Ax - b \Vert}{\Vert b \Vert}$', fontsize = font_size)
    # ax.set_ylabel(r'$\mathsf{Residual}$', fontsize = font_size)
    ax.set_yscale('log')    

    # Set font size for ticks
    ax.tick_params(axis='both', labelsize=font_size)

    # Custom legend
    # handles, labels = ax.get_legend_handles_labels()

    last_color = (len(list_iters)-1)/len(list_iters)
    #Custon legend de bernardo:
    legend_elements = [
    Line2D([0], [0], color=cmap(0), lw=2, label=r'$i_N = 1$'),
    Line2D([0], [0], ls='none', color='k', lw=2, label=r'', marker=r'$\cdot$', ms=5),
    Line2D([0], [0], ls='none', color='k', lw=2, label=r'', marker=r'$\cdot$', ms=5),
    Line2D([0], [0], ls='none', color='k', lw=2, label=r'', marker=r'$\cdot$', ms=5),
    Line2D([0], [0], color=cmap(last_color), lw=2, label=fr'$i_N = {list_iters[-1]}$'),
    ]
    ax.legend(handles=legend_elements, labelspacing=0.001, fontsize=font_size)
    plt.subplots_adjust(wspace=5)
    plt.tight_layout()
    os.makedirs('Imgs', exist_ok = True)
    plt.savefig(f'Imgs/errors.png', dpi = 300)
    plt.close()

def pre_post_sol_project(iN, i_gmres = 0):
    print(f'iN: {iN}, i_gmres: {i_gmres}')
    path = f'sol_proj/iN{iN:02}'
    if i_gmres:
        path += f'/i_gmres{i_gmres:02}'
    f_pre = np.load(path+'/fields_pre_sol_proj.npz')
    f_post = np.load(path+'/fields_post_sol_proj.npz')

    types = ['vx', 'vz', 'th']
    X_pre = []
    X_post = []
    for type_ in types:
        data_pre = f_pre[type_]
        data_post = f_post[type_]
        X_pre.append(data_pre)
        X_post.append(data_post)

        dif = data_pre - data_post
        # print(f'Pre sol.proj: |{type_}| = {np.linalg.norm(data_pre)} ')
        # print(f'Post sol.proj: |{type_}| = {np.linalg.norm(data_post)} ')
        # print(f'|{type_}_post - {type_}_pre|=  {np.linalg.norm(data_post-data_pre)} ')
    
    X_pre = np.concatenate(X_pre, axis = 1)
    X_post = np.concatenate(X_post, axis = 1)
    print(f'|X_pre| = {np.linalg.norm(X_pre)}')
    print(f'|X_post| = {np.linalg.norm(X_post)}')
    print(f'|X_post - X_pre| = {np.linalg.norm(X_post - X_pre)}')

def plot_orbit(iN):
    shape = (pm.Nx, pm.Ny, pm.Nz)
    os.makedirs('Imgs/output', exist_ok = True)

    types = ["th", "vx", "vz"]
    extent = [0, 2*np.pi, 0, 3.14]

    n_plots = 8 #amount of frames to plot
    #get # files
    files = os.listdir(f"output/iN{iN:02}/")
    n_files = len(files)//(len(types)+2) #because of pr and vy    


    # nmbs = list(range(0, n_fields, step)).append(n_fields)
    nmbs = GetSpacedElements(list(range(0, n_files)), n_plots)

    
    for type_data in types:
        fig, axs = plt.subplots(2, n_plots//2, figsize=(20, 10))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            file_ = f"output/iN{iN:02}/{type_data}.{nmbs[i]:05}.out"
            data = np.fromfile(file_, dtype=np.float64).reshape(shape,order='F')            
        
            cax = ax.imshow(data[:,0,:].T, extent=extent, vmin = -2., vmax = 2.)
            t = nmbs[i]*pm.dt*pm.ostep
            ax.set_title(f'{type_data}. t={round(t,2)} step={nmbs[i]}')
        # cbar = plt.colorbar(cax, ax = ax)
        cbar = fig.colorbar(cax, ax=axs,shrink=0.5, orientation = 'horizontal')#, pad = .02)

        plt.savefig(f'Imgs/output/{type_data}.{iN:02}.png', dpi = 300)
        plt.close() 

def hookstep(n_newt=None):
    dir_ = 'prints/hookstep/'
    if not n_newt:
        files = os.listdir(dir_)
        n_newt = len(files)//2

    iters = np.zeros(n_newt-1)
    for i in range(1, n_newt):
        data = np.loadtxt(f'{dir_}iN{i:02}.txt', delimiter = ',', skiprows = 1) #, unpack = True) 
        if data.ndim == 1:
            data = data[np.newaxis,:]

        iter_hook = data[:,0]
        if np.isscalar(iter_hook):
            iter_hook = np.array([iter_hook])

        iters[i-1] = iter_hook.shape[0]

        # iters[i-1] = iters_hook

    plt.figure()
    plt.plot(iters, marker = 'o')
    plt.xlabel(r'$i_N$')
    plt.ylabel(r'$i_{\mathsf{hook}}$')
    os.makedirs('Imgs', exist_ok = True)
    plt.savefig('Imgs/hookstep.png', dpi = 300)
    plt.close()


def main():
    # Check that a nmb was provided as a command-line argument
    if len(sys.argv) == 2:
        #  sys.exit(1)
        nmb = int(sys.argv[1])
    else:
        nmb = 1

    n_newt = 8 #amount of newton iterations completed

    # plot_errors()
    # hookstep()

    iN = 5
    # plot_diagnostic(iN)
    # plot_balance(iN)
    # balance_vs_init(iN)
    plot_orbit(iN)

    position =  pm.Nx//2, pm.Nz//2
    # plot_point(iN, position)


if __name__ == '__main__':
    main()