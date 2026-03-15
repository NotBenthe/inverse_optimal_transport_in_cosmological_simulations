import numpy as np
import pandas as pd

M_c = 10.0
expon = False

for n_boxes in [2, 4, 8]:
    for exp in [1]:

        iteration = "M" + str(int(M_c))
        if expon:
            iteration += "-exp" + str(exp)

        standpath = "./results/tests_with_motion_new/"
        begpath = "uniform_v_"
        #begpath = "nonuniform_v_"
        #begpath = "uniform_expansion_"
        #begpath = "uniform_collapse_"
        #begpath = "nonuniform_expansion_"
        #begpath = "nonuniform_collapse_"
        #begpath = "walls_"
        #begpath = "filament_"


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
                    chat = -epsilon * np.log(ratio) - epsilon

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

        if expon:
            pathpi = standpath + begpath + "pi-nbox" + str(n_boxes) + "-1000000-exp" + str(exp) + ".csv"
            pathmunu = standpath + begpath + "munu-nbox" + str(n_boxes) + "-1000000-exp" + str(exp) + ".csv"
        else:
            pathpi = standpath + begpath + "pi-nbox" + str(n_boxes) + "-1000000.csv"
            pathmunu = standpath + begpath + "munu-nbox" + str(n_boxes) + "-1000000.csv"

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

        # print(f"\nalpha: {alpha}\n\nbeta: {beta}\n\nc: {c}")

        filenamepi = standpath + begpath + "ma_n" + str(n_boxes) + "_c-" + str(iteration) + "-new.csv" 
        with open(filenamepi, 'w') as f:
            np.savetxt(f, c, delimiter=',')

        filenamemunu = standpath + begpath + "ma_n" + str(n_boxes) + "_ab-" + str(iteration) + "-new.csv" 
        with open(filenamemunu, 'w') as f:
            np.savetxt(f, np.column_stack((alpha, beta)), delimiter=',')
            print(f"{filenamepi} written to csv succesfully")
