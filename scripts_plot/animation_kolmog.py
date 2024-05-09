
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
# from matplotlib import rc
# rc('animation', html='jshtml')



T = 2
dt = 1e-3
save_freq = 50
nx = ny = 256
Nt = int(T/dt //save_freq)

uu_t = np.zeros((Nt,nx,ny))
vv_t = np.zeros((Nt,nx,ny))

path = '/share/data4/jcullen/pySPEC/run1/'
for i in range(Nt):
    step = save_freq * (1 + i)
    f_name = f'fields_{step:06}'
    fields = np.load(f'{path}output/{f_name}.npz')
    uu = fields['uu']
    vv = fields['vv']
    uu_t[i,:,:] = uu
    vv_t[i,:,:] = vv

print('Saved fields')

def get_var_name(variable):
    """devuelve el nombre de la variable (para usar en plt.title)
    """
    globals_dict = globals()

    return [
        var_name for var_name in globals_dict
        if globals_dict[var_name] is variable
    ][0] 

#plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\joaco\ffmpeg-6.0-full_build\bin"

def make_anim(u, cant_frames = 60,t_end = -1, dist_y = 0.1):
  """
  u: debe ser una matriz que en cada columna contenga el valor de u(x) para un tiempo, y x debe ir entre 0 y 2pi
  t_end: si querés animar hasta cierto tiempo le pasas el argumento t_end (tiene que ser el índice, no el tiempo)
  dist_y: si querés reducir la distancia entre el valor máximo (o mínimo) de la función y el valor máximo (o mín) del eje y
  """

  y_min = np.min(u)-dist_y
  y_max = np.max(u) + dist_y
  # First set up the figure, the axis, and the plot element we want to animate
  fig = plt.figure()
  ax = plt.axes(xlim=(-0.1, 2*np.pi+0.1), ylim=(y_min,y_max))
  line, = ax.plot([], [], lw=2)
  ax.set_xlabel("x")
  ax.set_ylabel("u")


  # initialization function: plot the background of each frame
  def init():
      line.set_data([], [])
      return line,

  # animation function.  This is called sequentially
  def animate(i):
      line.set_data(x, u[:,i])
      ax.set_title(titles_list[i])
      return line,

  #Cantidad de pasos temporales
  step = len(u[0,:t_end])
#  plt.xlim(-1, 1030)
  titles_list = [f"t = {round(i, 2)}" for i in np.arange(step)*dt]
  # call the animator.  blit=True means only re-draw the parts that have changed.
  anim = FuncAnimation(fig, animate, init_func=init,
                                frames=range(0,step, step//cant_frames), interval=100, blit=True)

  plt.close()
  return anim

def anim(V,label, save = False):
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

    # Create a figure and axis
    fig, ax = plt.subplots(figsize = (10,5))

    # Create the initial plot (null velocity field)
    im = ax.imshow((V[0,:,:].T), origin ='lower',animated=True)
    plt.colorbar(im)

    # Define a list of titles for each frame
    tot_frames = len(V[:,0,0])
    step_frames = 5
    t_step = save_freq*dt
    # turnover_t = 5.32
    #En turnover times 
    # title_list = [f'{get_var_name(V)} en {round(frame*t_step/turnover_t ,1)} turnover time (frame {frame})' for frame in range(tot_frames)]
    #En unidades de tiempo
    title_list = [f'{label} en t {round(frame*t_step ,1)} (frame {frame})' for frame in range(tot_frames)]

    # Create the animation
    anim_V = FuncAnimation(fig, update, frames=range(0,tot_frames, step_frames), interval=100)

    if save:
        writergif = animation.PillowWriter(fps=30) 
        filename = f'{path}plots/anims/{label}.gif'
        anim_V.save(filename, writer=writergif)
        # # Specify the filename and writer for saving
        # filename = f'{path}plots/anims/{label}.mp4'
        # # Save the animation
        # anim_V.save(filename, writer='ffmpeg', dpi=300, fps=10)
        # print('Saved ', label)

    # Display the animation
    
anim(uu_t, 'u(x,y)', save =True)
# plt.show()
# anim(vv_t, 'v(x,y)', save = False)
# plt.show()
