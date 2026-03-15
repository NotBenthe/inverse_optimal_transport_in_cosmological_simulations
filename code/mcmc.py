import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

#begpaths = ["nonuniform_expansion_", "nonuniform_collapse_"]
begpaths = ["nonuniform_expansion_"]

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

# defining prior and stuff
def log_prior(alpha):
    if np.any(alpha < 0):
        return -np.inf
    if np.any(np.abs(alpha) > 1e3):  # simple bound to keep sampler stable
        return -np.inf
    return -0.5 * np.sum((alpha / 2.0)**2)  # Gaussian prior centered ad 0 and sigma 2

def log_likelihood(alpha):
    mu = A_stack.dot(alpha)        # linear combination predicted T_flat
    resid = T_flat - mu
    return -0.5 * np.sum((resid / noise_std)**2) \
           - N_data * np.log(noise_std * np.sqrt(2*np.pi))  # log-likelihood

def log_probability(alpha):
    lp = log_prior(alpha)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(alpha)

# set up the properties of the problem.
ndim, nwalkers, nsampler = len(A_list), 2000, 1000

# initial guess: all matrices are working together at the same rate
pos = np.random.uniform(0.0001, 0.5, size=(nwalkers, ndim))
#theta_guess = np.zeros(ndim)
#pos = [theta_guess + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]

# Create the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

tmp = sampler.run_mcmc(pos, nsampler)

sampler.chain.shape

cutoff = 100

"""
fig, axes = plt.subplots(ncols=3, nrows=1)
fig.set_size_inches(13,6)
plt.rcParams.update({'font.size': 15})
axes[0].plot(sampler.chain[:, :, 0].transpose(), color='black', alpha=0.3)
axes[0].axvline(cutoff, ls='dashed', color='red')
axes[1].plot(sampler.chain[:, :, 1].transpose(), color='black', alpha=0.3)
axes[1].axvline(cutoff, ls='dashed', color='red')
axes[2].plot(sampler.chain[:, :, 2].transpose(), color='black', alpha=0.3)
axes[2].axvline(cutoff, ls='dashed', color='red')
fig.show()
"""

samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))
samples.shape

data_to_save = []
for k in range(len(samples[0])):
    summ = np.percentile(samples[:, k], [16, 50, 84])
    data_to_save.append({
        'k_value': k,
        'percentile_16': summ[0],
        'percentile_50': summ[1],
        'percentile_84': summ[2]
    })
    
df = pd.DataFrame(data_to_save)
csv_filename = 'exponly.csv'
df.to_csv(csv_filename, index=False)

print("lower bound, mean, upper bound")
for k in range(len(samples[0])):
    summ = np.percentile(samples[:, k], [16, 50, 84])
    print(f"{k}: {summ}")

#fig = corner.corner(samples, truths=[1, 1, 1],
#                    quantiles=[0.160, 0.50, 0.840], show_titles=True, title_fmt=.3)
#font = {'size'   : 11}
#plt.rc('font', **font)
#fig.savefig("cornerplot2.pdf")

