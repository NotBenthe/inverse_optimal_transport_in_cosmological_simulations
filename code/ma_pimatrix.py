import numpy as np
import pandas as pd
import swiftsimio as sw

from finals.boxes import divide_data_boxes
from finals.boxes import make_pi

n_boxes = 2

path0 = "/net/hypernova/data2/FLAMINGO/L0400N0360/DMO/flamingo_0000.hdf5"
#path0 = "../data/flamingo_0000.hdf5"

data = sw.load(path0)
ids = data.dark_matter.particle_ids
coordinates = data.dark_matter.coordinates * data.metadata.a
boxsize = data.metadata.boxsize * data.metadata.a

ids = np.array(ids)
boxsize = np.array(boxsize)
p0 = np.array(coordinates)
zips0 = np.column_stack((p0, ids))

data_boxes0 = divide_data_boxes(zips0, n_boxes, boxsize)

path8 = "/net/hypernova/data2/FLAMINGO/L0400N0360/DMO/flamingo_0008.hdf5"
#path8 = "../data/flamingo_0008.hdf5"

data = sw.load(path8)
ids = data.dark_matter.particle_ids
coordinates = data.dark_matter.coordinates * data.metadata.a
boxsize = data.metadata.boxsize * data.metadata.a

ids = np.array(ids)
boxsize = np.array(boxsize)
p8 = np.array(coordinates)
zips8 = np.column_stack((p8, ids))

data_boxes8 = divide_data_boxes(zips8, n_boxes, boxsize)

n = len(data_boxes0)
pi, mu, nu = make_pi(data_boxes0, data_boxes8, n)

filenamepi = "./results/actual_ma/pi-n" + int(n_boxes) + ".csv"  
with open(filenamepi, 'w') as f:
    np.savetxt(f, pi, delimiter=',', fmt='%d')

filenamemunu = "./results/actual_ma/munu-n" + int(n_boxes) + ".csv" 
with open(filenamemunu, 'w') as f:
    np.savetxt(f, np.column_stack((mu, nu)), delimiter=',', fmt='%d')
print(f"{filenamepi} written to csv succesfully")

M_c = 10

def ma_discrete(pi, mu, nu, n, max_iter=int(1e8), mins=1e-4, epsilon=1.0):
    # ensure mu, nu strictly positive (avoid divide-by-zero in updates)
    mu = mu.copy()
    nu = nu.copy()
    small = 1e-24
    mu[mu == 0] = small
    nu[nu == 0] = small
    mu = mu / np.sum(mu)
    nu = nu / np.sum(nu)

    n_dim = len(mu)
    # initial dual variables (not strictly necessary for masked recovery)
    a = np.random.rand(n_dim)
    b = np.random.rand(n_dim)

    # initial cost guess
    c = np.random.rand(n_dim, n_dim)

    # initialize Sinkhorn scalings
    u = np.exp(a)
    v = np.exp(b)

    # pre-allocate mask for cost recovery
    mask = (pi > 0)

    for counter in range(max_iter):
        # Sinkhorn kernel with safe clipping
        logK = np.clip(-c / epsilon, -100, 100)
        K_sinkhorn = np.exp(logK)

        # Sinkhorn updates
        u = mu / (K_sinkhorn @ v)
        v = nu / (K_sinkhorn.T @ u)

        # Recover cost only on support
        c_new = np.full_like(c, fill_value=M_c)
        
        # compute ratio and cost on pi>0
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = (u.reshape(-1, 1) * v.reshape(1, -1))
            ratio = pi[mask] / denom[mask]
            chat = -epsilon * np.log(ratio)

        # clamp and assign
        c_new[mask] = np.clip(chat, 0.0, M_c)

        # check convergence
        diff = np.linalg.norm(c_new - c)
        if counter % 10000 == 0:
            print(f"Iteration {counter}, cost change = {diff:.4e}")
        if diff < mins:
            print(f"Converged after {counter} iterations with diff={diff:.4e}")
            c = c_new
            break

        c = c_new

    # compute final dual potentials
    alpha = epsilon * np.log(u)
    beta = epsilon * np.log(v)
    return alpha, beta, c

n_boxes = 8
pathpi = filenamepi
pathmunu = filenamemunu

pi = pd.read_csv(pathpi, header=None)
munu = pd.read_csv(pathmunu, header=None)

pi = np.array(pi)
munu = np.array(munu)
mu = munu[:, 0]
nu = munu[:, 1]

n = float(360 ** 3)

pi = pi / n
mu = mu / n
nu = nu / n

alpha, beta, c = ma_discrete(pi, mu, nu, n, epsilon=1e-3)

filenamepi = "./actual_ma/c-n" + int(n_boxes) + ".csv" 
with open(filenamepi, 'w') as f:
    np.savetxt(f, c, delimiter=',')

filenamemunu = "./actual_ma/ab-n" + int(n_boxes) + ".csv" 
with open(filenamemunu, 'w') as f:
    np.savetxt(f, np.column_stack((alpha, beta)), delimiter=',')
print(f"{filenamepi} written to csv succesfully")
