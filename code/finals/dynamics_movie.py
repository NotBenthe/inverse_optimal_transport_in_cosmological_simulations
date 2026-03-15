import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from finals.read_data import get_data


def minimum_dists_eucl(p, R_0, box):
    """
    Calculates the minimum distance between two particles in a repeating box
    (vectorized version)

    p: coordinates of particles in format [[x1, y1, z1], [x2, y2, z2], ...]
    R_0: coordinates of the center point [x0, y0, z0]
    box: width of the box
    """
    diff = np.abs(p - R_0)
    alt_diff = np.abs(box - diff)
    min_diff = np.minimum(diff, alt_diff)
    distances = np.sqrt(np.sum(min_diff**2, axis=1))
    return distances


def ids_sphere(p, R_0, R, box):
    """ 
    Function to get the particle ids of all particles within a sphere of radius R centered at R_0

    p: coordinates of the particles (x, y, z)
    R_0: center of the sphere
    R: radius of the sphere
    box: length of the boxsize
    """ 
    dist = minimum_dists_eucl(p, R_0, box) 
    indices = np.where(dist <= R)[0]
    return indices


def moviemaker (steps_path, R, R_0, save_path = "", a = False, full=True):
    """ 
    Function to create a movie of timesteps of the simulations

    steps_path: filenames of all simulation timesteps
    R: radius
    R_0: midpoint of sphere of radius R
    save_path: save path. If left blank the movie will not save but show
    a: True multiplicates the data with the scale factor
    """

    # getting the first positions and indices of particles within sphere
    first, boxsize = get_data(steps_path[0], box=True, a=a)
    indices = ids_sphere(first, R_0, R, boxsize)

    # getting the positions of the particles in all timesteps
    p = []
    for path in steps_path:
        temp = get_data(path, a=a, target_ids=indices)
        p.append(temp)

    # creating figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # compute axis limits
    all_positions = np.vstack(p)
    xlims, ylims, zlims = [], [], []
    if full:
        xlims = [0, 400]
        ylims = [0, 400]
        zlims = [0, 400]
    else: 
        xlims = [np.min(all_positions[:, 0]), np.max(all_positions[:, 0])]
        ylims = [np.min(all_positions[:, 1]), np.max(all_positions[:, 1])]
        zlims = [np.min(all_positions[:, 2]), np.max(all_positions[:, 2])]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    # plotting initial positions 
    scat = ax.scatter(p[0][:, 0], p[0][:, 1], p[0][:, 2], c='blue', marker='o')

    def update(frame):
        """
        function to update the frame of the animation
        
        frame: frame number.
        """
        # resetting the plot
        ax.cla()  
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        
        # plotting the new timestip
        scat = ax.scatter(p[frame][:, 0], p[frame][:, 1], p[frame][:, 2], c='blue', marker='o')
        return scat

    # creating and saving animation
    n_frames = len(p)
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=500, blit=False)
    if save_path != "":
        ani.save(save_path + ".mp4", writer="ffmpeg", fps=2)
    plt.show()


def moviemaker_interpolate (steps_path, R, R_0, n_frames = 100, save_path = "", a = False, full = True):
    """ 
    Function to create a movie of the first and last positions, linearly interpolating in between

    steps_path: filenames of all simulation timesteps
    R: radius
    R_0: midpoint of sphere of radius R
    n_frames: number of frames
    save_path: save path. If left blank the movie will not save but show
    a: True multiplicates the data with the scale factor
    """

    def interpolate_positions(first, last, t):
        """
        Function to interpolate between first and last positions

        first: initial positions 
        last: final positions 
        t: time (between 0 and 1)
        """
        return first + t * (last - first)
    
    assert len(steps_path) == 2

    # getting the first positions and indices of particles within sphere
    first, boxsize = get_data(steps_path[0], box=True, a=a)
    indices = ids_sphere(first, R_0, R, boxsize)
    first = first[indices]
    last = get_data(steps_path[1], a=a, target_ids=indices)

    # creating the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # setting boundaries
    all_positions = np.vstack((first, last))
    xlims, ylims, zlims = [], [], []
    if full:
        xlims = [0, 400]
        ylims = [0, 400]
        zlims = [0, 400]
    else: 
        xlims = [np.min(all_positions[:, 0]), np.max(all_positions[:, 0])]
        ylims = [np.min(all_positions[:, 1]), np.max(all_positions[:, 1])]
        zlims = [np.min(all_positions[:, 2]), np.max(all_positions[:, 2])]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    # plotting the initial conditions
    scat = ax.scatter(first[:, 0], first[:, 1], first[:, 2], c='blue', marker='o')

    def update(frame):
        """
        function to update the frame of the animation
        
        frame: frame number.
        """
        t = frame / (n_frames - 1)  
        # interpolating the positions for the current frame
        position = interpolate_positions(first, last, t)  

        # Clear axis and reset limits and title (optional)
        ax.cla()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)

        scat = ax.scatter(position[:, 0], position[:, 1], position[:, 2], c='blue', marker='o')
        return scat

    # creating the animation and either showing or saving
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    if save_path != "":
        ani.save(save_path + ".mp4", writer="ffmpeg", fps=2)
    plt.show()
