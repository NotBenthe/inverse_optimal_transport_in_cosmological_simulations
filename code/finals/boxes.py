import numpy as np 

def get_boxes(n_boxes, maxs, mins = [0, 0, 0]):
    assert n_boxes > 0
    assert isinstance(n_boxes, int)
    
    totals = np.subtract(maxs, mins) 
    ranges = []
    for j in [0, 1, 2]:
        dx = totals[j]/n_boxes
        x_ranges = np.zeros(n_boxes+1)
        x_ranges[j] = mins[j]
        for i in range(1, n_boxes+1):
            x_ranges[i] = mins[j] + i * dx
        ranges.append(x_ranges)
    
    return np.array(ranges)


def get_boxes_multiple(n_boxes, maxs, mins = [0, 0, 0]):
    ranges = get_boxes(n_boxes, maxs, mins)
    all_ranges = []
    for j in [0, 1, 2]:
        x = []
        for i in range(len(ranges[j])-1):
            x.append([ranges[j][i], ranges[j][i+1]])
        all_ranges.append(x)
    return np.array(all_ranges)


def get_boxes_all(n_boxes, maxs, mins = [0, 0, 0]):
    all_ranges = get_boxes_multiple(n_boxes, maxs, mins)
    x = all_ranges[0]
    y = all_ranges[1]
    z = all_ranges[2]

    all_boxes = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                all_boxes.append([x[i], y[j], z[k]])
    return np.array(all_boxes)


def point_in_box(point, box):
    return (box[0][0] <= point[0] <= box[0][1] and
            box[1][0] <= point[1] <= box[1][1] and
            box[2][0] <= point[2] <= box[2][1]) 


def divide_data_boxes (data, n_boxes, maxs, mins = [0, 0, 0]):
    boxes = get_boxes_all(n_boxes, maxs, mins)

    points_boxes = {i: [] for i in range(len(boxes))}
    for point in data:
        for i, box in enumerate(boxes):
            if point_in_box(point, box):
                points_boxes[i].append(point)
                break
    return points_boxes


def make_pi(old_dict, new_dict, n):
    pi = np.zeros((n, n))
    mu = np.zeros(n)
    nu = np.zeros(n)
    old = {}
    new = {}

    for key, value in old_dict.items():
        mu[key] = len(value)
        for row in value:
            particle_id = int(row[3])
            old[particle_id] = key

    for key, value in new_dict.items():
        nu[key] = len(value)
        for row in value:
            particle_id = int(row[3])
            new[particle_id] = key
    
    for particle_id in old:
        if particle_id in new:
            i = old[particle_id]
            j = new[particle_id]
            pi[i, j] += 1

    return pi, mu, nu