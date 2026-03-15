import numpy as np

from finals.boxes import divide_data_boxes
from finals.boxes import make_pi

def get_points (center = [25, 25, 25], radius = 25, n_points = int(1e6), ids_add = True):
    points = (np.random.rand(n_points, 3) - 0.5) * radius * 2 + center
    ids = []
    if ids_add:
        ids = np.arange(1, n_points + 1)
    else:
        ids = np.zeros(n_points)
    points = np.column_stack((points, ids))
    return points 

n_boxes = 16
n_points = 1e6

# getting points for non-uniform motion
start = get_points(n_points = int(n_points))
end = start + get_points(n_points = int(n_points), ids_add=False)
boxsize = [100, 100, 100]

start = np.array(start)
end = np.array(end)
boxsize = np.array(boxsize)

start = divide_data_boxes(start, n_boxes, boxsize)
end = divide_data_boxes(end, n_boxes, boxsize)

n = len(start)
pi, mu, nu = make_pi(start, end, n)

filenamepi = "./results/tests_with_motion_new/nonuniform_v_pi-nbox" + str(n_boxes) + "-" + str(int(n_points)) + ".csv" 
with open(filenamepi, 'w') as f:
    np.savetxt(f, pi, delimiter=',')

filenamemunu = "./results/tests_with_motion_new/nonuniform_v_munu-nbox" + str(n_boxes) + "-" + str(int(n_points)) + ".csv" 
with open(filenamemunu, 'w') as f:
    np.savetxt(f, np.column_stack((mu, nu)), delimiter=',')
    print(f"{filenamepi} written to csv succesfully")