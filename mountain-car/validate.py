# Libraries
import numpy as np
import pickle as pkl
from statsSupplement import integrate_dirichlet
import os
import sys

# Load intervals
PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
with open(f"{PATH}/data/sample-intervals.pkl", "rb") as f:
    DATA = pkl.load(f)
lower_bounds = DATA["lower_bounds"]
upper_bounds = DATA["upper_bounds"]

# Load trajectories (switch between test and train data)
with open(f"{PATH}/data/sample-data.pkl", "rb") as f:
    DATA = pkl.load(f)
all_estimate_data = np.reshape(DATA["estimate_data_train"], (-1, 2))
all_state_data = DATA["state_data_train"]
total_samples = len(all_state_data)

# State space
sx = 0.005
x_space = [-1.2, 0.6]
sv = 0.001
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

# Loop through instances of validation experiments
num_instances = 10
all_confs = []
print(f"Starting {num_instances} instances")
for k in range(num_instances):

    ## Split data (70-30)
    num_out_samples = int(np.floor(total_samples * 0.7))
    out_indices =  np.random.choice(total_samples, size=num_out_samples, replace=False)
    state_data = all_state_data[out_indices]
    estimate_data = all_estimate_data[out_indices]

    # Loop through all state-space bins; aggregate eestimates into a set of intervals
    print(f"Starting validation loop for instance {k}...")
    alpha = 0.05 / (num_bins_err) # Bonferroni correction
    confidences = []
    for i in range(len(xbar_space)):
        xbin = xbar_space[i]

        # Structure out-of-domain data
        state_group = [s for s in state_data if (xbin*sx<=s[0]<(xbin*sx)+sx)]
        estimate_group = [est for s, est in zip(state_data, estimate_data) if (xbin*sx<=s[0]<(xbin*sx)+sx)]
        state_group = np.array(state_group)
        estimate_group = np.array(estimate_group)

        # Load in interval bounds
        lowers = lower_bounds[i]
        uppers = upper_bounds[i]

        # Compute conformance confidences
        if len(state_group) != 0:
            error_group_out = state_group[:, 0] - estimate_group[:, 0]
            counts, _ = np.histogram(error_group_out, bins=edges)
            N = counts.sum()
            K = len(counts)
            if N/K < 5: # data avilability threshold (increase if you run into problems)
                continue
            n_total = len(error_group_out) # total number of samples
            n_total_in = N # total samples that fall into the bins
            n_total_out = n_total - n_total_in # total number of samples that fall out of the bins

            # Initialize prior parameters with ones; adjust totals accordingly
            counts = counts + np.ones_like(counts)

            # Compute freqs relative to real_total
            in_freqs = counts
            out_freq = n_total_out + 1

            # Set up dirichlet parameters
            support = [(l, u) for l, u in zip(lowers.tolist(), uppers.tolist())]
            support.append((0.0, 1.0))
            alphas = in_freqs.tolist()
            alphas.append(int(out_freq)) # list parameter is the out-of-range freq
            
            # Integrate dirichlet over support
            conf, _ = integrate_dirichlet(alphas, support)
            confidences.append(conf)
            print(f"    Out-of-domain confidence at {i}: {conf} over {n_total} samples")

    # Append collected confidences to externl list
    all_confs.append(confidences)

    # Save validations confidence results
    print(f"Wrapping up instance {k}...")
    print(f"%========= Results of instance {k} =========%")
    print(f"Out-of-domain domain data:")
    print(f"  o Mean: {np.mean(confidences)}")
    print(f"  o Min: {np.min(confidences)}")
    print(f"  o Max: {np.max(confidences)}")
    print(f"  o Median: {np.median(confidences)}")
    print(f"Continuing to next instance...")
print("Process finished.")
