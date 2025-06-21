# Libraries
import numpy as np
import pickle as pkl
from scipy.stats import beta
from plant import SyntheticSystem
import os
import sys

PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path

# State space abstraction parameters
sx = 0.5
x_space = [0.0, 12.0]
sy = 0.5
y_space = [0.0, 12.0]

# Estimation space abstraction parameters
sxhat = 1.0
xhat_space = [8.0, 12.0]
syhat = 1.0
yhat_space = [8.0, 12.0]
x_edges = [8.0, 9.0, 10.0, 11.0, 12.0]
y_edges = [8.0, 9.0, 10.0, 11.0, 12.0]

# Define discrete space
xbar_space = list(range(int(x_space[0]/sx), int(x_space[1]/sx)))
ybar_space = list(range(int(y_space[0]/sy), int(y_space[1]/sy)))
xhatbar_space = list(range(int(xhat_space[0]/sxhat), int(xhat_space[1]/sxhat)))
yhatbar_space = list(range(int(yhat_space[0]/syhat), int(yhat_space[1]/syhat)))
xhat_min, xhat_max = xhat_space
yhat_min, yhat_max = yhat_space
num_bins_xhat = len(xhatbar_space)
num_bins_yhat = len(yhatbar_space)

# Preallocate relevant data
lower_bounds = np.empty((len(xbar_space), len(ybar_space)), dtype=object)
upper_bounds = np.empty((len(xbar_space), len(ybar_space)), dtype=object)
num_state_points = np.zeros((len(xbar_space), len(ybar_space)))

# Instantiate system
args = {
    "bias": [0.0, 0.0],
    "cov_lb": [[0.5, 0],[0, 0.5]],
    "cov_ub": [[2.0, 0],[0, 2.0]],
    "init_state": [0.0, 0.0],
    "goal_state": [10.0, 10.0],
    "gain": 0.5,
    "max_step": 0.7,
    "success_dist": 2.0,
    "time_limit": np.inf,
    "barricades": [
        # {"x_min": 0.0, "x_max": 15.0, "y_min": 12.0, "y_max": 15.0}
    ]}
clsys_train = SyntheticSystem(args)

# Loop through all state-space bins; aggregate estimates into a set of intervals
alpha = 0.01 / (num_bins_xhat * num_bins_yhat) # Bonferroni correction
np.random.seed(23)
confidences = []
for i in range(len(xbar_space)):
    xbar = xbar_space[i]
    for j in range(len(ybar_space)):
        ybar = ybar_space[i]
        num_samples = 200
        estimate_group = []
        for k in range(num_samples):
            xhat = np.random.uniform(xbar*sx, (xbar*sx)+sx)
            yhat = np.random.uniform(ybar*sy, (ybar*sy)+sy)
            clsys_train.state = np.array([xhat, yhat], dtype=float)
            state_est, _, _, _, _ = clsys_train.step()
            estimate_group.append(state_est)
        estimate_group = np.array(estimate_group)

        # Construct Binomial confidence intervals
        counts, xedges, yedges = np.histogram2d(estimate_group[:, 0],
                                                estimate_group[:, 1],
                                                bins=[x_edges, y_edges])
        counts = counts + np.ones_like(counts) 
        counts = counts.flatten() 
        N = counts.sum()
        lowers = []
        uppers = []
        for ni in counts:
            lowers.append(beta.ppf(alpha / 2, ni, N - ni + 1))
            uppers.append(beta.ppf(1 - alpha / 2, ni + 1, N - ni))
        lowers = np.array(lowers)
        uppers = np.array(uppers)
        lowers = np.reshape(lowers, (num_bins_xhat, num_bins_yhat))
        uppers = np.reshape(uppers, (num_bins_xhat, num_bins_yhat))

        # Allocate upper/lower bounds to external array
        lower_bounds[i, j] = lowers
        upper_bounds[i, j] = uppers

# Save confidence intervals
DATA = {
    "lower_bounds": lower_bounds,
    "upper_bounds": upper_bounds
}
with open(f"{PATH}/data/sample-intervals.pkl", "wb") as f:
    pkl.dump(DATA, f)
