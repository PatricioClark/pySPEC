import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.time_marching import SWHD_1D

from matplotlib.animation import ArtistAnimation
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# path where to find parameters
param_path = 'examples/time_marching_swhd1D_DG_hb_noise-scaled'
# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open(f'{param_path}/params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.Lx

# define plot function for gifs
def plot_time_lapse(field,
                    num_arrays,
                    Lx,
                    Nx,
                    name='animation.gif',
                    color='brown',
                    label='field',
                    limit = False,
                    interval=200,
                    fps=5):
    '''A function that animates a list of arrays for a field of (x, t)'''

    array_length = len(field[0])
    domain = np.linspace(0, Lx, Nx)

    # Create a list to store frames
    frames = []

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlabel('x')

    # Create the initial frame
    line, = ax.plot(domain, field[0], color=color, label=label)
    if limit:
        ax.set_ylim(field[0].min()*(1+0.1) , field[0].max()*(1+0.1))
    frames.append([line])

    for i, d in enumerate(field):
        if i >= num_arrays:
            break
        # Update the line for each frame
        line.set_ydata(d)
        frames.append([ax.plot(domain, d, color=color)[0]])

    ax.legend()

    # Create the animation
    ani = ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation as a GIF
    ani.save(name, writer='pillow', fps=fps)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# Define plot function for gifs with two fields
def plot_time_lapse_two_fields(field1,
                               field2,
                               num_arrays,
                               Lx,
                               Nx,
                               name='animation.gif',
                               color1='brown',
                               color2='blue',
                               label1='field1',
                               label2='field2',
                               limit=False,
                               interval=200,
                               fps=5):
    '''A function that animates two lists of arrays for two fields (x, t)'''

    domain = np.linspace(0, Lx, Nx)
    frames = []

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlabel('x')

    # Initialize the lines for the two fields
    line1, = ax.plot(domain, field1[0], color=color1, label=label1)
    line2, = ax.plot(domain, field2, color=color2, label=label2)

    # Set y-limits if specified
    if limit:
        combined_min = min(field1[0].min(), field2.min()) * (1 + 0.1)
        combined_max = max(field1[0].max(), field2.max()) * (1 + 0.1)
        ax.set_ylim(combined_min, combined_max)

    # Append the initial lines
    frames.append([line1, line2])

    # Create frames for the animation
    for i, d1 in enumerate(field1):
        if i >= num_arrays:
            break
        # Update the lines for each frame
        line1.set_ydata(d1)
        line2.set_ydata(field2)
        frames.append([ax.plot(domain, d1, color=color1)[0],
                       ax.plot(domain, field2, color=color2)[0]])

    ax.legend()

    # Create the animation
    ani = ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation as a GIF
    ani.save(name, writer='pillow', fps=fps)

# load frames
steps = int(pm.T/pm.dt) # total steps of the simulation
U = np.load(f'{pm.out_path}/uu_memmap.npy')[::pm.plot_step]
H = np.load(f'{pm.out_path}/hh_memmap.npy')[::pm.plot_step]
hb = np.load(f'{pm.out_path}/hb.npy')
# plot gifs for u and h and save to out_path
plot_time_lapse(U ,steps, Lx = pm.Lx , Nx = pm.Nx,
                name = f'{param_path}/u.gif',
                    color = 'blue' ,
                    label = '$u(x,t)$',
                    interval = 200 ,
                    fps = 5 )
plot_time_lapse(H ,steps, Lx = pm.Lx , Nx = pm.Nx,
                name = f'{param_path}/h.gif',
                    color = 'green' ,
                    label = '$h(x,t)$',
                    interval = 200 ,
                    fps = 5 )
plot_time_lapse_two_fields(H,
                               hb,
                               steps,
                               Lx = pm.Lx,
                               Nx = pm.Nx,
                               name = f'{param_path}/h_hb.gif',
                               color1='brown',
                               color2='blue',
                               label1='$h(x,t)$',
                               label2='$h_b(x)$',
                               limit=False,
                               interval=200,
                               fps=5)
