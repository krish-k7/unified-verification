# Libraries
import numpy as np
import pickle as pkl
from scipy.stats import beta
import os
import sys

# Load mountaincar training data
PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
with open(f"{PATH}/data/sample-data.pkl", "rb") as f:
    DATA = pkl.load(f)
estimate_data_train = DATA["estimate_data_train"]
state_data_train = DATA["state_data_train"]

# State space
sx = 0.05
x_space = [-1.2, 0.6]
sv = 0.01
v_space = [-0.07, 0.07]

# Estimation space
serr = 0.1
err_space = [-0.2, 0.4]
edges = np.array([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4], dtype=float)

# Define discrete space
xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)
vbar_space = np.arange(int(v_space[0]/sv), int(v_space[1]/sv)+1)
errbar_space = np.arange(int(err_space[0]/serr), int(err_space[1]/serr))
err_min, err_max = min(err_space), max(err_space)
num_bins_err = len(errbar_space)

# Preallocate relevant data
lower_bounds = []
upper_bounds = []
num_state_points = []

# Loop through all position bins; aggregate estimates into a set of intervals indexed by positions
alpha = 0.01 / (num_bins_err) # Bonferroni correction
confidences = []
for i in range(len(xbar_space)):
    xbin = xbar_space[i]

    # Structure and allocate training data by bins
    state_group = [s for s in state_data_train if (xbin*sx<=s[0]<(xbin*sx)+sx)]
    estimate_group = [est for s, est in zip(state_data_train, estimate_data_train) if (xbin*sx<=s[0]<(xbin*sx)+sx)]
    state_group = np.array(state_group)
    estimate_group = np.array(estimate_group)

    if len(estimate_group) != 0:

        # Construct Binomial confidence intervals
        error_group = state_group[:, 0] - estimate_group[:, 0]
        counts, _ = np.histogram(error_group, bins=edges)
        counts = counts + np.ones_like(counts) # condition to prevent nan uppers
        N = counts.sum()
        K = len(counts)
        if N/K > 5:
            lowers = []
            uppers = []
            for ni in counts:
                lowers.append(beta.ppf(alpha / 2, ni, N - ni + 1))
                uppers.append(beta.ppf(1 - alpha / 2, ni + 1, N - ni))
            lowers = np.array(lowers)
            uppers = np.array(uppers)
        else: # insufficent data exists to make claims about distribution
            lowers = np.zeros((len(errbar_space)))
            uppers = np.ones((len(errbar_space)))
    else:
        N = 0 # No state data exists in this bin, so CIs cannot be constructed
        lowers = np.zeros((len(errbar_space)))
        uppers = np.ones((len(errbar_space)))

    # Allocate upper/lower bounds to external array
    num_state_points.append(N)
    lower_bounds.append(lowers)
    upper_bounds.append(uppers)

    # print(f"Computed uppers: {uppers}")

lower_bounds = np.array(lower_bounds)
upper_bounds = np.array(upper_bounds)

print(np.shape(lower_bounds))

DATA = {
    "lower_bounds": lower_bounds,
    "upper_bounds": upper_bounds
}
with open(f"{PATH}/data/sample-intervals.pkl", "wb") as f:
    pkl.dump(DATA, f)
