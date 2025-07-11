# Libraries
import numpy as np
from prismgen import generate_prism_model
from genIntervals import generate_intervals
from pathlib import Path
import sys
import os
import subprocess
import re

PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # path to the file folder
SUPER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
STUDY_PATH = f"{PATH}/data/ablation-data"

# Reusable paths
intpath = f"{STUDY_PATH}/tmp-intervals.pkl" # path to the intervals file
modelpath = f"{STUDY_PATH}/imdp.pm" # path to save the prism model
proppath = f"{PATH}/prism-models/synth-props.props" # path to the MC PCTL properties for any prism model
prismpath = f"{SUPER_PATH}/prism_ws/bin/prism"

# Reusable parameters
system_args = {
    "bias": [0.0, 0.0],
    "cov_lb": [[0.5, 0],[0, 0.5]],
    "cov_ub": [[2.0, 0],[0, 2.0]],
    "init_state": [0.0, 0.0],
    "goal_state": [10.0, 10.0],
    "gain": 0.5,
    "max_step": 0.7,
    "success_dist": 2.0,
    "time_limit": np.inf,
    "barricades": []
}
sx = 0.5
x_space = [0.0, 12.0]
sy = 0.5
y_space = [0.0, 12.0]
sxhat = 1.0
xhat_space = [8.0, 12.0]
syhat = 1.0
yhat_space = [8.0, 12.0]
cudd_mem = 4 # memory for CUDD in GB
java_mem = 4 # memory for Java in GB

# Constant interval generation hyperparameters for the ablation
int_args = {}
int_args['sx'] = sx
int_args['x_space'] = x_space
int_args['sy'] = sy
int_args['y_space'] = y_space
int_args['sxhat'] = sxhat
int_args['xhat_space'] = xhat_space
int_args['syhat'] = syhat
int_args['yhat_space'] = yhat_space
int_args['num_samples'] = 200
int_args['system_args'] = system_args
int_args['savepath'] = intpath
int_args['alpha'] = 0.001 # constant alpha for this ablation
print(f"Generating intervals with constant alpha = 0.001...")
generate_intervals(int_args)

# Initial states to test
initial_states = [
    [0.0, 0.0],
    [0.0, 2.5],
    [2.5, 0.0],
    [2.5, 2.5],
    [0.0, 5.0],
    [5.0, 0.0],
    [5.0, 5.0],
    [0.0, 7.5],
    [7.5, 0.0],
    [7.5, 7.5],
]

# Preallocate safety chances
beta = []

# Loop through initial states; generate prism model; run verification
for state in initial_states:

    # Abstract initial state
    init_state = [int(np.floor(state[0] / sx)), int(np.floor(state[1] / sy))] # abstract initial state

    # Verification arguments
    ver_args = {}
    ver_args['sx'] = sx
    ver_args['x_space'] = x_space
    ver_args['sy'] = sy
    ver_args['y_space'] = y_space
    ver_args['sxhat'] = sxhat
    ver_args['xhat_space'] = xhat_space
    ver_args['syhat'] = syhat
    ver_args['yhat_space'] = yhat_space
    ver_args['system_args'] = system_args
    ver_args['intpath'] = intpath
    ver_args['init_state'] = init_state
    ver_args['modelpath'] = modelpath

    # Generate PRISM model
    print(f"Generating PRISM model with initial state = {state}...")
    generate_prism_model(ver_args)

    # Run verification
    print(f"Running verification subprocess for initial state = {state}...")
    term_cmd = [prismpath, "-cuddmaxmem", f"{cudd_mem}g", "-javamaxmem", f"{java_mem}g", modelpath, proppath]
    result = subprocess.run(term_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pmin = float(re.search(r"Result:\s*([0-9.]+)", result.stdout).group(1))
    beta.append(pmin)
    print("    >", re.search(r"^Time for model construction:.*$", result.stdout, re.MULTILINE).group(0))
    print("    >", f"Minimum safety chance: {pmin}")

# Print summary of results
print("\nSummary of Results:")
for state, pmin in zip(initial_states, beta):
    print(f"Initial state {state}: Minimum safety chance = {pmin}")
    