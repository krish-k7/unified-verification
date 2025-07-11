# Libraries
import numpy as np
from genIntervals import generate_intervals
from getData import get_data
from validate import run_validation
import os

# Locate path to save data and intervals
PATH = os.path.dirname(os.path.abspath(__file__))
STUDY_PATH = f"{PATH}/data/ablation-data"

# Reusable paths
intpath = f"{STUDY_PATH}/tmp-intervals.pkl"  # path to save the intervals file
testpath = f"{STUDY_PATH}/test-data.pkl"  # path to save the test data

# Reusable parameters
sx = 0.5  # state space resolution
x_space = [0.0, 12.0]  # state space range
sy = 0.5  # error space resolution
y_space = [0.0, 12.0]  # error space range
sxhat = 1.0  # estimate space resolution
xhat_space = [8.0, 12.0]  # estimate space range
syhat = 1.0  # estimate error space resolution
yhat_space = [8.0, 12.0]  # estimate error space range

# Generate intervals with no bias
interval_args = {
    'system_args': {
            "bias": [0.0, 0.0],
            "cov_lb": [[0.5, 0.0], [0.0, 0.5]],
            "cov_ub": [[2.0, 0.0], [0.0, 2.0]],
            "init_state": [0.0, 0.0],
            "goal_state": [10.0, 10.0],
            "gain": 0.5,
            "max_step": 0.7,
            "success_dist": 2.0,
            "time_limit": 25,
            "barricades": []
        },
    'sx': sx,
    'x_space': x_space,
    'sy': sy,
    'y_space': y_space,
    'sxhat': sxhat,
    'xhat_space': xhat_space,
    'syhat': syhat,
    'yhat_space': yhat_space,
    'alpha': 0.001,
    'num_samples': 200,
    'savepath': intpath
}
print("Generating intervals with no bias...")
generate_intervals(interval_args)

# List of biases to test
dists = [0.0, 2.45, 2.47, 2.50, 2.52, 2.55, 2.6, 2.7, 2.8, 2.9, 3.0]

# Loop through biases and validate
all_results = []
for dist in dists:

    print(f"Testing with offset = {dist}...")
    bias_x = dist / np.sqrt(2)
    bias_y = -dist / np.sqrt(2)

    # Generate test data with the current bias
    data_args = {
        'system_args': {
            "bias": [bias_x, bias_y],
            "cov_lb": [[0.5, 0.0], [0.0, 0.5]],
            "cov_ub": [[2.0, 0.0], [0.0, 2.0]],
            "init_state": [0.0, 0.0],
            "goal_state": [10.0, 10.0],
            "gain": 0.5,
            "max_step": 0.7,
            "success_dist": 2.0,
            "time_limit": 25,
            "barricades": [
                {"x_min": 0.0, "x_max": 15.0, "y_min": 12.0, "y_max": 15.0},
                {"x_min": 12.0, "x_max": 15.0, "y_min": 0.0, "y_max": 15.0}
            ]
        },
        'num_traj': 1000,
        'alpha': 0.01,
        'savepath': testpath
    }
    get_data(data_args)

    # Validate against the generated intervals
    validation_args = {
        'sx': sx,
        'x_space': x_space,
        'sy': sy,
        'y_space': y_space,
        'sxhat': sxhat,
        'xhat_space': xhat_space,
        'syhat': syhat,
        'yhat_space': yhat_space,
        'num_instances': 1,
        'testpath': testpath,
        'intpath': intpath,
        'display': False
    }
    confs = run_validation(validation_args)
    median_conf = np.median(confs[0])
    print(f"    > Median confidence: {median_conf:.4f}")
