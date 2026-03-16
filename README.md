# Inverse Optimal Transport in Cosmological Simulations

The theory of Optimal Transport (OT) provides a powerful mathematical framework for understanding dynamics, with applications in various disciplines. In cosmology, OT can offer a unique view through which to analyze structure formation and evolution, by connecting initial and final density configurations.  

In this thesis, we investigate the structure formation in cosmology using an inverse OT approach, focusing on the movement of particles. We utilize data from the FLAMINGO simulations, segmented into smaller boxes for computational efficiency. Our primary objective was to infer the cost matrix that is responsible for the particle motion within these simulations. We developed the theory of inverse OT and developed an algorithm to solve for the cost matrix, while addressing the ill-posed nature of inverse OT. This algorithm is applied to both reference cases and the FLAMINGO simulation data. The derived cost matrix shows a high degree of symmetry, consistent with an isotropic universe. As a proof of concept, a Markov Chain Monte Carlo simulation was done to test if the FLAMINGO cost matrix is a linear combination of reference cases. These Markov chains did not converge. However, this is thought to be because of the limited number of reference cases. 

This thesis is submitted as final thesis for a Bachelors in Mathematics and Astronomy, under supervision by Dr. M. Schaller, Dr. E. Sellentin and Dr. D. van der Hoeven. The final thesis can be found ![here](/inverse_optimal_transport_in_cosmological_simulations_Benthe_Sturre.pdf). 

## Requirements
```
numpy
scipy
swiftsimio
matplotlib
velociraptor
VirgoDC
LightconeIO
warnings
h5py
numba
unyt
PIL
itertools
pandas
cvxpy
torch
ot
time
random
pylab
cv2
csv
mpl_toolkits
itertools
```

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── code 						# all the code
	└── finals 					# code that gets imported into other files
		└── boxes.py			# dividing the data up into boxes
		└── cost_euclidian.py	# cost function
		└── dynamics_movie.py	# making a movie of the simulation dynamics
		└── ma2021.py			# doing the Ma 2021 algorithm
		└── minimize_cost.py	# minimizing the cost function
		└── read_data.py		# reading the data
	└── box_for_thesis.ipynb	# making of figure 3.2
	└── example_sinkhorn.ipynb	# example of Sinkhorn calculation
	└── filaments.py			# calculating the cost matrix for filaments
	└── finalmcmc.ipynb			# MCMC code
	└── linear_combination.py	# heatmap of MCMC linear combination of cost matrix 
	└── ma.py					# running the ma algorithm
	└── mcmc.py					# MCMC code
	└── nonuniform_collapse.py	# calculating the cost matrix for nonuniform collapse
	└── nonunifom_expansion.py	# calculating the cost matrix for nonuniform expansion
	└── nonuniform_v.py			# calculating the cost matrix for nonuniform velocity
	└── uniform_collapse.py		# calculating the cost matrix for uniform collapse
	└── uniform_expansion.py	# calculating the cost matrix for uniform expansion
	└── uniform_v.py			# calculating the cost matrix for uniform velocity
	└── visualization.ipynb		# vizualization code
	└── walls.py				# calculating the cost matrix for walls
└── README.md         			# This file
└── requirements.txt 			# required modules for the code
└── thesis    					# thesis TEX files
	└── images    				# thesis images/results
└── inverse_optimal_transport_in_cosmological_simulations_Benthe_Sturre.pdf 	# final thesis
```


