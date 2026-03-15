import numpy as np
from scipy.optimize import minimize

def calculate_pihat (mu, nu):
    """ 
    Function to calculate pi_hat
    
    mu, nu: numpy arrays of initial distribution, of the form [[x1, y1, z1], ....]
    
    returns pi_hat with pi_hat_{(i,k), (j,m)} = mu_{i,k} nu_{j,m}
    """
    mu = np.array(mu)
    nu = np.array(nu) 
    
    # calculating the length of mu, nu
    mu_rows, mu_cols = mu.shape
    nu_rows, nu_cols = nu.shape
    
    # initiating pihat
    pihat = np.zeros((mu_rows, mu_cols, nu_rows, nu_cols))
    
    # calculating each index of pihat
    for i in range(mu_rows):
        for k in range(mu_cols):
            for j in range(nu_rows):
                for m in range(nu_cols):
                    pihat[i, k, j, m] = mu[i, k] * nu[j, m]
    return pihat 


def R (x):
    # return np.linalg.norm(x)
    return 0
    # return np.sum(x**2)


def ma_discrete(pi, mu, nu, max_iter=1000, mins=1e-9, epsilon=1.0, gamma=1.0):
    """ 
    Beginnings of a function to do the inverse OT
    
    pi: transport matrix (array in R^(NxN))
    mu, nu: initial and final distribution (array in R^N)
    max_iter: maximum number of iterations
    mins: difference needed for convergence
    dim: dimension of the problem
    epsilon: as in ma2021
    gamma: as in ma2021
    
    returns the Lagrangian multipliers and cost matrix
    """

    dim = len(mu)
    
    # initializing the Lagrangian multipliers
    a = np.ones(dim) 
    b = np.ones(dim)
    
    # initial guess for the cost matrix
    c = np.ones((dim, dim))
    
    # calculating u, v 
    u = np.exp(a / epsilon)
    v = np.exp(b / epsilon)
    
    for counter in range(max_iter):
        # going through the algorithm: 
        K = np.exp(-c / epsilon)
        u_new = mu / np.matmul(K, v)
        v_new = nu / np.matmul(np.transpose(K), u_new)
        K = pi / np.matmul(u_new, np.transpose(v_new))
        
        # making chat to use in the optimization
        chat = -epsilon * np.log(K)
        
        def argmin_function(c_flat):
            c_matrix = c_flat.reshape((dim, dim))
            return np.sum(R(c_matrix) + (1 / (2 * gamma)) * np.linalg.norm(c_matrix - chat)**2)
        
        # minimizing the prox_gamma
        result = minimize(argmin_function, c.flatten()) 
        c_new = result.x.reshape((dim, dim)) 
        
        # see if things converge
        if np.linalg.norm(c_new - c) < mins:
            print(f"break\niterations = {counter}")  # prints the amount of iterations needed before convergence
            break
        
        # trying again with new values
        u, v, c = u_new, v_new, c_new
    
    # setting alpha, beta to their true values
    alpha = epsilon * np.log(u)
    beta = epsilon * np.log(v)
    
    return alpha, beta, c



def savetxt(p, filename):
    with open(filename, 'w') as f:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                for k in range(p.shape[2]):
                    for l in range(p.shape[3]):
                        f.write(f"{p[i, j, k, l]} ")
                    f.write("\n")
                f.write("\n")
            f.write("\n")



def ma_discrete_3d(pi, mu, nu, max_iter=1000, mins=1e-9, epsilon=1.0, gamma=1.0):
    """ 
    Beginnings of a function to do the inverse OT
    
    pi: transport matrix (array in R^(3xNx3xN))
    mu, nu: initial and final distribution (array in R^3xN)
    max_iter: maximum number of iterations
    mins: difference needed for convergence
    dim: dimension of the problem
    epsilon: as in ma2021
    gamma: as in ma2021
    
    returns the Lagrangian multipliers and cost matrix
    """

    dim = len(mu)

    # initializing the Lagrangian multipliers and the cost matrix
    a = np.random.rand(dim, 3)
    b = np.random.rand(dim, 3)
    c = np.random.rand(dim, 3, dim, 3) * 10

    # savetxt(c, "./results/20250403_datatest_ma2021-c.txt")

    #a = np.ones((dim, 3)) 
    #b = np.ones((dim, 3))
    #c = np.ones((dim, 3, dim, 3))
    
    # calculating u, v 
    u = np.exp(a / epsilon)
    v = np.exp(b / epsilon)
    
    for counter in range(max_iter):
        # going through the algorithm: 
        K = np.exp(-c / epsilon)
        u_new = mu / np.tensordot(K, v, axes=([2, 3], [0, 1]))
        v_new = nu / np.tensordot(K.transpose((2, 3, 0, 1)), u_new, axes=([2, 3], [0, 1]))
        K = pi / np.tensordot(u_new, v_new, axes=0)
        
        # making chat to use in the optimization
        chat = -epsilon * np.log(K)
        
        def argmin_function(c_flat):
            c_matrix = c_flat.reshape((dim, 3, dim, 3))
            return np.sum(R(c_matrix) + (1 / (2 * gamma)) * np.linalg.norm(c_matrix - chat)**2)

        # minimizing the prox_gamma
        result = minimize(argmin_function, c.flatten(), method='L-BFGS-B')
        c_new = result.x.reshape((dim, 3, dim, 3)) 
        
        # see if things converge
        if np.linalg.norm(c_new - c) < mins:
            print(f"break\niterations = {counter}")  # prints the amount of iterations needed before convergence
            break
        
        # trying again with new values
        u, v, c = u_new, v_new, c_new
    
    # setting alpha, beta to their true values
    alpha = epsilon * np.log(u)
    beta = epsilon * np.log(v)
    
    return alpha, beta, c
