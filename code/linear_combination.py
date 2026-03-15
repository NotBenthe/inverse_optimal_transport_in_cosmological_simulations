import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

font = {'size'   : 16}
matplotlib.rc('font', **font)

rad = 100
moveto = [rad, rad, rad, 0]

def generate_combinations(val = 25, max = 375, radius = 50):
    nums = []
    while val <= max:
        nums.append(val)
        val += radius

    combs = []
    for x in nums:
        for y in nums:
            for z in nums:
                combs.append([x, y, z])

    return combs

centers = generate_combinations(val = rad, max = 400 - moveto[0], radius=moveto[0])

standpath = "./ref_cases/"

begpaths = ["nonuniform_expansion_", "nonuniform_collapse_"]

A_list = []

for begpath in begpaths:
    for exp in [0.4, 1, 2, 5]:
        for center in centers:
            pathpi = standpath + begpath + "ma-c-center" + str(center) + "-rad100-move100exp-" + str(exp) + ".csv"
            c = pd.read_csv(pathpi, header=None)
            c = np.array(c)
            A_list.append(c)

n_vars = len(A_list)
guess = np.random.rand(n_vars)

pathtrue = "./actual_ma/c-n8.csv"
T_true = pd.read_csv(pathtrue, header=None)
T_true = np.array(T_true)
noise_std = 0.5

# flatten for vector computations
A_stack = np.stack([M.flatten() for M in A_list], axis=1) 
T_flat  = T_true.flatten()
N_data  = T_flat.size

mcmc_results_df = pd.read_csv('mcmcexpcoll.csv')
alpha_median = mcmc_results_df['percentile_50'].values

# Calculate the linear combination using the median alpha values
# This is equivalent to mu = A_stack.dot(alpha) from your log_likelihood function
T_predicted_flat = A_stack.dot(alpha_median)

# Reshape the predicted flat array back to the original T_true shape
# Assuming T_true had a shape, you can get it from T_true.shape
T_predicted = T_predicted_flat.reshape(T_true.shape)

n_boxes = 8
zeros = True
m_no = 10

c = T_predicted

if zeros:
    c[c == m_no] = -5
    c[c == -5] = np.max(c)

plt.figure(figsize=(8,6))
plt.imshow(c, cmap="viridis")
plt.colorbar(pad=0.1)
plt.ylabel("i")
plt.xlabel("j")
plt.gca().invert_yaxis()
pathc = "./mcmcexpcoll"
plt.title("heatmap of MCMC linear combination of cost matrix " + r"$c_{ij}$" + f" with n = {n_boxes}")
plt.savefig(pathc + "-zeros.pdf")
print(f"saved as {pathc + ".pdf"}")
plt.tight_layout()
plt.show()