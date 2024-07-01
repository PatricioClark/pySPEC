
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
# from matplotlib import rc
#plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\joaco\ffmpeg-6.0-full_build\bin"
# rc('animation', html='jshtml')



T = 2
dt = 1e-3
save_freq = 50
nx = ny = 256
Nt = int(T/dt //save_freq)

path = '/share/data4/jcullen/pySPEC/run1/'
# path = 'C:/Users/joaco/Desktop/FCEN/Tesis/Patrick/pySPEC/'
fields = np.load(f'{path}output2/fields_t.npz')
uu_t = fields['uu_t']
vv_t = fields['vv_t']
print('Loaded fields')


def get_var_name(variable):
    """devuelve el nombre de la variable (para usar en plt.title)
    """
    globals_dict = globals()

    return [
        var_name for var_name in globals_dict
        if globals_dict[var_name] is variable
    ][0] 


j = 0
def anim(V,label,t_sta,t_end, cant_frames, save = False):
    """
    Le ingreso un array 4D con t en primera coord y devuelve evol temporal.
    """
    # Function to update the plot for each frame
    def update(frame):
        # Clear the current plot
        plt.clf()
        # Plot the data
        plt.imshow((V[frame,:,:].T), vmin = np.min(V[:,:,:]), vmax = np.max(V[:,:,:]), origin ='lower',animated=True)
        cbar = plt.colorbar()
        cbar.set_label(f'{label}')
        # Set the title with a varying value
        plt.title(title_list[frame])
    # Define a list of titles for each frame
    t_step = save_freq*dt
    ind_sta,ind_end = int(t_sta/t_step),int(t_end/t_step) 
    tot_frames = ind_end-ind_sta
    frame_list = [frame for frame in range(ind_sta, ind_end,tot_frames//cant_frames)]
    # step_frames = 5
    #En unidades de tiempo
    title_list = [f'{label} en t {round(frame*t_step ,1)} (frame {frame})' for frame in range(0, ind_end)]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize = (10,5))

    # Create the initial plot (null velocity field)
    im = ax.imshow((V[ind_sta,:,:].T), origin ='lower',animated=True)
    plt.colorbar(im)
    # Create the animation
    anim_V = FuncAnimation(fig, update, frames=frame_list, interval=100)

    if save:
        # writergif = animation.PillowWriter(fps=30) 
        # filename = f'{path}plots/anims/{label}.gif'
        # anim_V.save(filename, writer=writergif)
        # # Specify the filename and writer for saving
        filename = f'{path}plots/anims/{label}.mp4'
        # Save the animation
        anim_V.save(filename, writer='ffmpeg', dpi=300, fps=10)
        print('Saved ', label)

    # Display the animation
    
# anim(uu_t, 'u(x,y)',0,499, int(499*20) , save =True)
# plt.show()
# anim(vv_t, 'v(x,y)', save = False)
# plt.show()


stat = 500_000
save_freq = 50
dt = 1e-3
def imshow(field, label, i):
    step = stat + save_freq * i
    t = round(step*dt, 2)
    plt.figure()
    plt.imshow(field.T, origin = 'lower', extent = [0, nx, 0, ny])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    title = f'{label}_t={t}({step})'
    plt.title(title)
    plt.savefig(f'plots/fields/{label}/{title}.png')
    plt.close()

for i in range(uu_t.shape[0]):
    uu, vv = uu_t[i,:,:], vv_t[i,:,:]
    imshow(uu, 'u', i)
    imshow(vv, 'v', i)
