# Libraries
import numpy as np
import pickle as pkl
from statsSupplement import integrate_dirichlet
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

# Define discrete space
xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx))
ybar_space = np.arange(int(y_space[0]/sy), int(y_space[1]/sy))
xhatbar_space = np.arange(int(xhat_space[0]/sxhat), int(xhat_space[1]/sxhat))
yhatbar_space = np.arange(int(yhat_space[0]/syhat), int(yhat_space[1]/syhat))
xhat_min, xhat_max = xhat_space
yhat_min, yhat_max = yhat_space
num_bins_xhat = len(xhatbar_space)
num_bins_yhat = len(yhatbar_space)

# Load data
with open(f"{PATH}/data/sample-intervals.pkl", "rb") as f:
    DATA = pkl.load(f)
lower_bounds = DATA["lower_bounds"]
upper_bounds = DATA["upper_bounds"]
with open(f"{PATH}/data/sample-data.pkl", "rb") as f:
    DATA = pkl.load(f)
state_data = DATA["state_data"]
estimate_data = DATA["estimate_data"]
total_data = len(state_data)

# Parameters
split_percent = 0.8
num_instances = 10

# Loop through instances of validation experiments
split_seeds = np.arange(1, num_instances+1, 1)*1
all_confs = []
print(f"Starting {num_instances} instances")
for k in range(num_instances):

    # Split data
    num_samples = int(np.floor(total_data * 0.5))
    indices = indices = np.random.choice(total_data, size=num_samples, replace=False)
    state_data_split = state_data[indices]
    estimate_data_split = estimate_data[indices]

    # Loop through all state-space bins; aggregate eestimates into a set of intervals
    print(f"Starting validation loop for instance {k}...")
    confidences = []
    for i in range(len(xbar_space)):
        xbin = xbar_space[i]
        for j in range(len(ybar_space)):
            ybin = ybar_space[j]

            # Structure in-domain data
            state_group = [s for s in state_data_split if ((xbin*sx<=s[0]<(xbin*sx)+sx) & (ybin*sy<=s[1]<(ybin*sy)+sy))]
            estimate_group = [est for s, est in zip(state_data_split, estimate_data_split) if ((xbin*sx<=s[0]<(xbin*sx)+sx) & (ybin*sy<=s[1]<(ybin*sy)+sy))]
            state_group = np.array(state_group)
            estimate_group = np.array(estimate_group)

            # Load in interval bounds
            lowers = lower_bounds[i, j]
            uppers = upper_bounds[i, j]

            # Compute conformance confidence of in domain data to these intervals
            if len(estimate_group) != 0:
                x_edges = np.linspace(xhat_min, xhat_max, num_bins_xhat + 1)
                y_edges = np.linspace(yhat_min, yhat_max, num_bins_yhat + 1)
                counts, xedges, yedges = np.histogram2d(estimate_group[:, 0],
                                                        estimate_group[:, 1],
                                                        bins=[x_edges, y_edges])
                n_total = counts.sum() # total within the specified range
                if n_total < 100: # data avilability threshold
                    continue
                real_total = len(estimate_group)
                n_total_out = real_total - n_total

                # Initialize prior parameters with ones; adjust totals accordingly
                counts += 10
                n_total += 160
                real_total += 160

                # Compute freqs relative to real_total
                in_freqs = counts
                out_freq = n_total_out + 1

                # Set up dirichlet parameters
                support = [(l, u) for l, u in zip(lowers.flatten().tolist(), uppers.flatten().tolist())]
                support.append((0.0, 1.0))
                alphas = in_freqs.flatten().tolist()
                alphas.append(float(out_freq)) # list parameter is the out-of-range freq
                
                # Integrate dirichlet over support
                conf, _ = integrate_dirichlet(alphas, support)
                confidences.append(conf)
                print(f"    Confidence at {(i, j)}: {conf} over {real_total} samples")

    # Append collected confidences to externl list
    all_confs.append(confidences)

    # Save validations confidence results
    print(f"Wrapping up instance {k}...")
    print(f"%========= Results of instance {k} =========%")
    print(f"In-domain data:")
    print(f"  o Mean: {np.mean(confidences)}")
    print(f"  o Min: {np.min(confidences)}")
    print(f"  o Max: {np.max(confidences)}")
    print(f"  o Median: {np.median(confidences)}")
    print(f"Continuing to next instance...")