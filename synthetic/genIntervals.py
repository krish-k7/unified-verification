# Libraries
import numpy as np
import pickle as pkl
from scipy.stats import beta
from plant import SyntheticSystem
import os
import sys

def generate_intervals(args):
    
    # Unpack arguments
    sx = args['sx']
    x_space = args['x_space']
    sy = args['sy']
    y_space = args['y_space']
    sxhat = args['sxhat']
    xhat_space = args['xhat_space']
    syhat = args['syhat']
    yhat_space = args['yhat_space']
    alpha = args['alpha']
    num_samples = args['num_samples']
    savepath = args['savepath']

    # Define discrete space
    xbar_space = list(range(int(x_space[0]/sx), int(x_space[1]/sx)))
    ybar_space = list(range(int(y_space[0]/sy), int(y_space[1]/sy)))
    xhatbar_space = list(range(int(xhat_space[0]/sxhat), int(xhat_space[1]/sxhat)))
    yhatbar_space = list(range(int(yhat_space[0]/syhat), int(yhat_space[1]/syhat)))
    num_bins_xhat = len(xhatbar_space)
    num_bins_yhat = len(yhatbar_space)
    x_edges = np.arange(xhat_space[0], xhat_space[1] + sxhat, sxhat, dtype=float)
    y_edges = np.arange(yhat_space[0], yhat_space[1] + syhat, syhat, dtype=float)

    # Preallocate relevant data
    lower_bounds = np.empty((len(xbar_space), len(ybar_space)), dtype=object)
    upper_bounds = np.empty((len(xbar_space), len(ybar_space)), dtype=object)

    # Instantiate system
    clsys_train = SyntheticSystem(args['system_args'])

    # Loop through all state-space bins; aggregate estimates into a set of intervals
    for i in range(len(xbar_space)):
        xbar = xbar_space[i]
        for j in range(len(ybar_space)):
            ybar = ybar_space[j]
            estimate_group = []
            for k in range(num_samples):
                xhat = np.random.uniform(xbar*sx, (xbar*sx)+sx)
                yhat = np.random.uniform(ybar*sy, (ybar*sy)+sy)
                clsys_train.state = np.array([xhat, yhat], dtype=float)
                state_est, _, _, _, _ = clsys_train.step()
                estimate_group.append(state_est)
            estimate_group = np.array(estimate_group)

            # Construct Binomial confidence intervals
            counts, _, _ = np.histogram2d(estimate_group[:, 0],
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
            lowers = np.array(lowers).reshape((num_bins_xhat, num_bins_yhat))
            uppers = np.array(uppers).reshape((num_bins_xhat, num_bins_yhat))

            # Allocate upper/lower bounds to external array
            lower_bounds[i, j] = lowers
            upper_bounds[i, j] = uppers

    # Save confidence intervals
    DATA = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds
    }
    with open(savepath, "wb") as f:
        pkl.dump(DATA, f)

# Sample usage
if __name__ == "__main__":
    
    # Locate path to save intervals
    PATH = os.path.dirname(os.path.abspath(sys.argv[0]))

    args = {
        'system_args': {
            "bias": [0.0, 0.0],
            "cov_lb": [[0.5, 0.0],[0.0, 0.5]],
            "cov_ub": [[2.0, 0.0],[0.0, 2.0]],
            "init_state": [0.0, 0.0],
            "goal_state": [10.0, 10.0],
            "gain": 0.5,
            "max_step": 0.7,
            "success_dist": 2.0,
            "time_limit": np.inf,
            "barricades": []
        },
        'sx': 0.5,
        'x_space': [0.0, 12.0],
        'sy': 0.5,
        'y_space': [0.0, 12.0],
        'sxhat': 1.0,
        'xhat_space': [8.0, 12.0],
        'syhat': 1.0,
        'yhat_space': [8.0, 12.0],
        'alpha': 0.01,
        'num_samples': 200,
        'savepath': f'{PATH}/data/sample-intervals.pkl',
    }
    print("Generating intervals...")
    generate_intervals(args)
    print("Intervals generated and saved to", args['savepath'])
