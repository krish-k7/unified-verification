# Libraries
import numpy as np
import pickle as pkl
from statsSupplement import integrate_dirichlet

def run_validation(args):

    # Unpack arguments
    sx = args['sx']  # state space resolution
    x_space = args['x_space']  # state space range
    serr = args['serr']  # error space resolution
    err_space = args['err_space']  # error space range
    num_instances = args['num_instances']  # number of validation instances
    testpath = args['testpath']  # path to the test data
    intpath = args['intpath']  # path to the intervals file
    display = args.get('display', False) # whether to display progress

    # Load intervals
    with open(intpath, "rb") as f:
        DATA = pkl.load(f)
    lower_bounds = DATA["lower_bounds"]
    upper_bounds = DATA["upper_bounds"]

    # Load test trajectories
    with open(testpath, "rb") as f:
        DATA = pkl.load(f)
    all_estimate_data = np.reshape(DATA["estimate_data_train"], (-1, 2))
    all_state_data = DATA["state_data_train"]
    total_samples = len(all_state_data)

    # Define discrete space
    xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)
    edges = np.arange(err_space[0], err_space[1] + serr, serr, dtype=float)

    # Loop through instances of validation experiments
    all_confs = []
    if display:
        print(f"Starting {num_instances} instances")
    for k in range(num_instances):

        ## Split data (70-30)
        num_out_samples = int(np.floor(total_samples * 0.7))
        out_indices =  np.random.choice(total_samples, size=num_out_samples, replace=False)
        state_data = all_state_data[out_indices]
        estimate_data = all_estimate_data[out_indices]

        # Loop through all state-space bins; aggregate eestimates into a set of intervals
        if display:
            print(f"Starting validation loop for instance {k}...")
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
                if display:
                    print(f"    > Confidence at {i}: {conf} over {n_total} samples")

        # Append collected confidences to external list
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
    if display:
        print("Process finished.")

    return all_confs

# Sampel usage
if __name__ == "__main__":

    # Find path to save intervals file
    import os
    import sys
    PATH = os.path.dirname(os.path.abspath(sys.argv[0]))

    args = {
        'sx': 0.05,
        'x_space': [-1.2, 0.6],
        'serr': 0.1,
        'err_space': [-0.2, 0.4], 
        'num_instances': 1,
        'testpath': f'{PATH}/data/sample-test-data.pkl',
        'intpath': f'{PATH}/data/sample-intervals.pkl',
        'display': True,
    }
    print("Starting validation process...\n")
    run_validation(args)  # Call the function to run validation
    print("Validation complete.")