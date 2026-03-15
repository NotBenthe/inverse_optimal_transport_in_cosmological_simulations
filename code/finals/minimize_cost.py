import numpy as np
import scipy as sc 

def minimize_cost (x, y, box, costfunction):
    """ 
        Function to minimize a given cost function

        x, y: datapoints from (x) to (y)
        box: boxsize
        costfunction: specified cost function, i.e. the Euclidian distance

        returns the minimum cost and the best permutation
    """
    assert len(x) == len(y)
    n = len(x)

    # generating a matrix with costs
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = costfunction(x[i], y[j], box)

    # getting the optimal row- and column indices for the permutation
    row_ind, col_ind = sc.optimize.linear_sum_assignment(cost_matrix)
    min_cost = cost_matrix[row_ind, col_ind].sum()
    best_permutation = col_ind.tolist()

    return min_cost, tuple(best_permutation)