import numpy as np
import matplotlib.pyplot as plt
import os

path = 'prints/hookstep'
os.makedirs('Imgs', exist_ok=True)

files = os.listdir(path)
files = [f for f in files if f.startswith('extra')]

#Newton completed iterations:
iNs = len(files)

def hook_ratio_all(show_last = False):
    Deltas = [] #trust region
    y_norms = [] #norm after hookstep
    ratios = [] #ratio between Delta and y_norm
    iN_ = [] #saves the iteration number for each pair (Delta,y_norm)

    idxs = [] #saves last idxs for each iN
    i = 0 #Counter for Delta idx

    for iN in range(1,iNs):
        file = np.loadtxt(f'{path}/extra_iN{iN:02}.txt', skiprows=1, delimiter=',', unpack=True)
        # Manage empty files
        if file.size==0:
            continue
        # Manage files with only one line
        Delta_,_,y_norm_,_ = file
        if isinstance(Delta_, np.float64):
            Delta_ = [Delta_]
            y_norm_ = [y_norm_]

        # Iterate
        for j, (Delta, y_norm) in enumerate(zip(Delta_,y_norm_)):
            if y_norm < Delta:
                Deltas.append(Delta)
                y_norms.append(y_norm)
                ratios.append(y_norm/Delta)
                iN_.append(iN)
                if show_last and j == len(Delta_)-1:
                    idxs.append(i)
                i += 1

    Deltas, ratios, iN_ = np.array(Deltas), np.array(ratios), np.array(iN_)
    plot(Deltas, ratios, iN_, title='hookstep_ratio', idxs = idxs)
    return 

def hook_ratio_last():
    Deltas = [] #trust region
    y_norms = [] #norm after hookstep
    ratios = [] #ratio between Delta and y_norm
    iN_ = [] #saves the iteration number for each pair (Delta,y_norm)


    for iN in range(1,iNs):
        file = np.loadtxt(f'{path}/extra_iter{iN:02}.txt', skiprows=1, delimiter=',', unpack=True)
        # Manage empty files
        if file.size==0:
            continue
        # Manage files with only one line
        Delta_,_,y_norm_,_ = file
        if isinstance(Delta_, np.float64):
            Delta_ = [Delta_]
            y_norm_ = [y_norm_]

        Delta, y_norm = Delta_[-1], y_norm_[-1]
        Deltas.append(Delta)
        y_norms.append(y_norm)
        ratios.append(y_norm/Delta)
        iN_.append(iN)

    Deltas, ratios, iN_ = np.array(Deltas), np.array(ratios), np.array(iN_)
    plot(Deltas, ratios, iN_, title='hookstep_ratio_last')
    return


def plot(Deltas, ratios, iN_, title, idxs=None):
    plt.figure()
    sc = plt.scatter(iN_, ratios, c=Deltas, cmap='viridis')
    plt.colorbar(sc, label='Delta')
    # Highlight last idxs
    if idxs:
        plt.scatter(iN_[idxs], ratios[idxs], c='red', s=4)
    plt.xlabel('Newton iteration')
    plt.ylabel('||y||/Delta')
    plt.savefig(f'Imgs/{title}.png', dpi = 300)
    plt.close()


show_last = True
hook_ratio_all(show_last = show_last)

# hook_ratio_last()
