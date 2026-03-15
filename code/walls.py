import numpy as np

from finals.boxes import divide_data_boxes
from finals.boxes import make_pi

n_boxes = 8
n_points = int(1e6)
for exp in [0.4, 1, 2, 5]:

    def get_points(center=[50, 50, 50], radius=25, collapse_exp=2.0, expand_exp=2.0, n_points=1_000_000, ids_add=True):
        starts = (np.random.rand(n_points, 3) - 0.5) * (2.0 * radius) + center
        deltas = starts - center
        abs_delta = np.abs(deltas)
        signs = np.sign(deltas)

        # Expansion in x and y
        extra_xy = radius * (abs_delta[:, :2] / radius) ** expand_exp
        end_abs_xy = abs_delta[:, :2] + extra_xy

        # Collapse in z
        extra_z = -radius * (abs_delta[:, 2] / radius) ** collapse_exp
        end_abs_z = np.clip(abs_delta[:, 2] + extra_z, 0.0, None)

        # Combine expanded xy and collapsed z
        end_abs = np.column_stack((end_abs_xy, end_abs_z))
        ends = center + signs * end_abs

        ids = np.arange(1, n_points + 1)
        start = np.column_stack((starts, ids))
        end = np.column_stack((ends, ids))
        return start, end

    # Generate the points
    start, end = get_points(center=[50, 50, 50], radius=25, collapse_exp=exp, expand_exp=exp, n_points=n_points)

    # Define boxsize and divide points into boxes
    boxsize = np.array([100, 100, 100])
    start = divide_data_boxes(start, n_boxes, boxsize)
    end = divide_data_boxes(end, n_boxes, boxsize)

    # Create transport plan
    n = len(start)
    pi, mu, nu = make_pi(start, end, n)

    # Save to files
    filenamepi = f"./results/tests_with_motion_new/walls_pi-nbox{n_boxes}-{n_points}-exp{exp}.csv"
    with open(filenamepi, 'w') as f:
        np.savetxt(f, pi, delimiter=',')

    filenamemunu = f"./results/tests_with_motion_new/walls_munu-nbox{n_boxes}-{n_points}-exp{exp}.csv"
    with open(filenamemunu, 'w') as f:
        np.savetxt(f, np.column_stack((mu, nu)), delimiter=',')

    print(f"{filenamepi} written to csv successfully")
