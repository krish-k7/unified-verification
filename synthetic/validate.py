# Libraries
import numpy as np
import pickle as pkl
from statsSupplement import integrate_dirichlet

def run_validation(args):

    # Unpack arguments
    sx = args['sx']  # state space resolution
    x_space = args['x_space']  # state space range
    sy = args['sy']  # error space resolution
    y_space = args['y_space']  # error space range
    sxhat = args['sxhat']  # estimate space resolution
    xhat_space = args['xhat_space']  # estimate space range
    syhat = args['syhat']  # estimate error space resolution
    yhat_space = args['yhat_space']  # estimate error space range
    num_instances = args['num_instances']  # number of validation instances
    testpath = args['testpath']  # path to the test data
    intpath = args['intpath']  # path to the intervals file
    display = args.get('display', False) # whether to display progress

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
    with open(intpath, "rb") as f:
        DATA = pkl.load(f)
    lower_bounds = DATA["lower_bounds"]
    upper_bounds = DATA["upper_bounds"]
    with open(testpath, "rb") as f:
        DATA = pkl.load(f)
    state_data = DATA["state_data"]
    estimate_data = DATA["estimate_data"]
    total_data = len(state_data)

    # Loop through instances of validation experiments
    all_confs = []
    if display:
        print(f"Starting {num_instances} instances")
    for k in range(num_instances):

        # Split data
        num_samples = int(np.floor(total_data * 0.5))
        indices = indices = np.random.choice(total_data, size=num_samples, replace=False)
        state_data_split = state_data[indices]
        estimate_data_split = estimate_data[indices]

        # Loop through all state-space bins; aggregate eestimates into a set of interval
        if display:
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
                    if display:
                        print(f"    > Confidence at {(i, j)}: {conf} over {real_total} samples")

        # Append collected confidences to externl list
        all_confs.append(confidences)

        # Save validations confidence results
        if display:
            print(f"Wrapping up instance {k}...")
            print(f"%========= Results of instance {k} =========%")
            print(f"    > Mean: {np.mean(confidences)}")
            print(f"    > Min: {np.min(confidences)}")
            print(f"    > Max: {np.max(confidences)}")
            print(f"    > Median: {np.median(confidences)}")
            print(f"Continuing to next instance...")

    return all_confs

# Sample usage
if __name__ == "__main__":

    # Locate path to load test data and intervals
    import os
    import sys
    PATH = os.path.dirname(os.path.abspath(sys.argv[0]))

    args = {
        'sx': 0.5,
        'x_space': [0.0, 12.0],
        'sy': 0.5,
        'y_space': [0.0, 12.0],
        'sxhat': 1.0,
        'xhat_space': [8.0, 12.0],
        'syhat': 1.0,
        'yhat_space': [8.0, 12.0],
        'num_instances': 1,
        'testpath': f'{PATH}/data/sample-test-data.pkl',
        'intpath': f'{PATH}/data/sample-intervals.pkl',
        'display': True,
    }
    print("Starting validation process...\n")
    run_validation(args)  # Call the function to run validation
    print("Validation complete.")
