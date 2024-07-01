import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

def get_var_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable][0] 

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

def get_T(nstep):
    solver = np.loadtxt('prints/solver.txt', delimiter = ',', skiprows = 1)
    return solver[nstep,1]

def mkdir_orb(orb_n):
    orbs = np.loadtxt('readme.txt', delimiter=',', skiprows=1, comments='-', usecols = (0,1,2))
    orb = orbs[orb_n]
    dir_name = f'orb{orb[0]}_{orb[1]}'
    mkdir(dir_name)

def copy_files(orb_n):
    orbs = np.loadtxt('readme.txt', delimiter=',', skiprows=1, comments='-', usecols = (0,1,2))
    orb = orbs[orb_n]
    dir_name = f'orb{int(orb[0]):2}_{orb[1]}'
    dir_name1 = 'orb19_15.9'
    subprocess.run(f'cp -r {dir_name1}/ {dir_name}/', shell = True)
    subprocess.run(f'cd {dir_name}', shell = True)
    # subprocess.run('clear_test', shell = True) #No me lo reconoce

n = 25
# mkdir_orb(n)
copy_files(n)