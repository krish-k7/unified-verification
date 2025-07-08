# Libraries
import numpy as np
import pickle as pkl
from scipy.stats import beta

def generate_intervals(args):

    # Unpack arguments
    sx = args['sx']  # state space resolution
    x_space = args['x_space']  # state space range
    serr = args['serr']  # error space resolution
    err_space = args['err_space']  # error space range
    alpha = args['alpha']  # confidence level for confidence intervals
    datapath = args['datapath']  # path to the training data
    savepath = args['savepath']  # path to the intervals file

    # Load mountaincar training data
    with open(datapath, "rb") as f:
        DATA = pkl.load(f)
    estimate_data_train = DATA["estimate_data_train"]
    state_data_train = DATA["state_data_train"]

    # Define discrete space
    xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)
    errbar_space = np.arange(int(err_space[0]/serr), int(err_space[1]/serr))
    num_bins_err = len(errbar_space)
    edges = np.arange(err_space[0], err_space[1] + serr, serr, dtype=float)

    # Preallocate relevant data
    lower_bounds = []
    upper_bounds = []
    num_state_points = []

    # Loop through all position bins; aggregate estimates into a set of intervals indexed by positions
    alpha = alpha / (num_bins_err) # Bonferroni correction
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

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    DATA = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds
    }
    with open(savepath, "wb") as f:
        pkl.dump(DATA, f)
