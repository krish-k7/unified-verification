# Libraries
from plant import SyntheticSystem
import numpy as np
import pickle as pkl
from scipy.stats import beta
import os
import sys

PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path

# Constructs clopper-pearson binomial confidence intervals
def clopper_pearson(k, n, alpha=0.01):
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")
    # lower bound
    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha/2, k, n - k + 1)
    # upper bound
    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha/2, k + 1, n - k)
    return lower, upper

# Instantiate system
delta_var = 0.0
dist = 0.0
bias_x = dist / np.sqrt(2)
bias_y = -dist / np.sqrt(2)
print(f"Bias in x: {bias_x}, Bias in y: {bias_y}")
args = {
    "bias": [0.0+bias_x, 0.0+bias_y],
    "cov_lb": [[0.5+delta_var, 0.0],[0.0, 0.5+delta_var]],
    "cov_ub": [[2.0+delta_var, 0.0],[0.0, 2.0+delta_var]],
    "init_state": [0.0, 0.0],
    "goal_state": [10.0, 10.0],
    "gain": 0.5,
    "max_step": 0.7,
    "success_dist": 2.0,
    "time_limit": 25,
    "barricades": [
        {"x_min": 0.0, "x_max": 15.0, "y_min": 12.0, "y_max": 15.0},
        {"x_min": 12.0, "x_max": 15.0, "y_min": 0.0, "y_max": 15.0}
    ]}
clsys = SyntheticSystem(args)

# Collect data from out-of-domain environment
num_traj = 1000
state_data = []
estimate_data = []
num_fail = 0
steps = []
for k in range(num_traj):

    states = []
    estimates = []
    time = 0
    while clsys.terminal == False:
        time += 1
        state = clsys.state.copy()
        states.append(state)
        s_hat, _, _, _, _ = clsys.step()
        estimates.append(s_hat)
    
    if clsys.failed:
        num_fail += 1
    steps.append(time)

    states = np.array(states)
    clsys.reset()
    state_data.extend(states)
    estimate_data.extend(estimates)
state_data = np.array(state_data)
estimate_data = np.array(estimate_data)

lower, upper = clopper_pearson(num_traj - num_fail, num_traj, alpha=0.01)
print(f"Success rate: {1-(num_fail / num_traj)}, interval: [{round(lower, 4)}, {round(upper, 4)}]")

DATA = {
    "state_data": state_data,
    "estimate_data": estimate_data
}
with open(f"{PATH}/data/sample-data.pkl", "wb") as f: # in-domain or out-of-domain data
    pkl.dump(DATA, f)
