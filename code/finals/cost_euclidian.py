import numpy as np

def minimum_dist_eucl (x, y, D):
    """ 
    Calculates the minimum distance between two particles in a repeating box

    p1, p2: coordinates of particles 1 and 2 in format [[x1, y1, z1], [x2, y2, z2], ...]
    D: width of the box
    """
    # asserting the points have the same dimensions
    assert len(x) == len(y)

    x1 = np.abs(np.subtract(x, y))
    x2 = np.abs(D - np.abs(np.subtract(x, y)))
    minimum = np.minimum(x1, x2)
    return np.sqrt(np.sum(minimum**2))

def cost_eucl(x, y, box):
    """ 
    Function to calculate the cost function:
    |x - y|^2 * rho_0(x)
    assuming rho_0(x) is constant and equal to 1
    """
    assert len(x) == len(y)
    dist = minimum_dist_eucl(x, y, box)
    cost = np.sum(np.sum(dist**2))

    return cost
