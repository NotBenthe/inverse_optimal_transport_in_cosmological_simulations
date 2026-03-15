import numpy as np

from finals.boxes import divide_data_boxes
from finals.boxes import make_pi

n_boxes = 16
n_points = 1e6

def get_points(center = [50, 50, 50], start_radius = 50, end_radius = 25, n_points = int(1e6), ids_add = True):
    start = (np.random.rand(n_points, 3) - 0.5) * start_radius * 2 + center
    initial_distances = start - center
    scale_factor = end_radius / start_radius
    final_dist = initial_distances * scale_factor
    points_end = center + final_dist

    ids = np.arange(1, n_points + 1)
    start = np.column_stack((start, ids))
    end = np.column_stack((points_end, ids))
    return start, end

start, end = get_points(center=[50, 50, 50], start_radius=50, end_radius=25)
boxsize = [100, 100, 100]

start = np.array(start)
end = np.array(end)
boxsize = np.array(boxsize)

start = divide_data_boxes(start, n_boxes, boxsize)
end = divide_data_boxes(end, n_boxes, boxsize)

n = len(start)
pi, mu, nu = make_pi(start, end, n)

filenamepi = "./results/tests_with_motion_new/uniform_collapse_pi-nbox" + str(n_boxes) + "-" + str(int(n_points)) + ".csv" 
with open(filenamepi, 'w') as f:
    np.savetxt(f, pi, delimiter=',')

filenamemunu = "./results/tests_with_motion_new/uniform_collapse_munu-nbox" + str(n_boxes) + "-" + str(int(n_points)) + ".csv" 
with open(filenamemunu, 'w') as f:
    np.savetxt(f, np.column_stack((mu, nu)), delimiter=',')
    print(f"{filenamepi} written to csv succesfully")