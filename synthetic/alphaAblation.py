# Libraries
import numpy as np
from genIntervals import generate_intervals
from getData import get_data
from validate import run_validation
from prismgen import generate_prism_model
from pathlib import Path
import sys
import os
import subprocess
import re

PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # path to the file folder
SUPER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
STUDY_PATH = f"{PATH}/data/ablation-data"

# Reusable paths
trainpath = f"{STUDY_PATH}/train-data.pkl" # path to save the training data
testpath = f"{STUDY_PATH}/test-data.pkl" # path to save the test data
intpath = f"{STUDY_PATH}/tmp-intervals.pkl" # path to save the intervals file
modelpath = f"{STUDY_PATH}/imdp.pm" # path to save the prism model
proppath = f"{PATH}/prism-models/synth-props.props" # path to the synthetic PCTL properties for any prism model
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

# Initial synthetic system state
init_state = [0.0, 0.0]
init_state = [int(np.floor(init_state[0] / sx)), int(np.floor(init_state[1] / sy))] # abstract initial state

# Generate training and validation data
data_args = {}
system_args['time_limit'] = 25
system_args['barricades'] = [{"x_min": 0.0, "x_max": 15.0, "y_min": 12.0, "y_max": 15.0},
                             {"x_min": 12.0, "x_max": 15.0, "y_min": 0.0, "y_max": 15.0}]
data_args['system_args'] = system_args
data_args['num_traj'] = 1000 # number of trials to run
data_args['alpha'] = 0.05 # note: this alpha is different from the alpha used in the IMDP intervals
data_args['savepath'] = testpath
print("Collecting test trajectories...")
get_data(data_args) # generate test data

# Reset system arguments
system_args['time_limit'] = np.inf
system_args['barricades'] = []

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

# Constant validation hyperparameters for the ablation
val_args = {}
val_args['sx'] = sx
val_args['x_space'] = x_space
val_args['sy'] = sy
val_args['y_space'] = y_space
val_args['sxhat'] = sxhat
val_args['xhat_space'] = xhat_space
val_args['syhat'] = syhat
val_args['yhat_space'] = yhat_space
val_args['num_instances'] = 1 # number of validation instances
val_args['testpath'] = testpath
val_args['intpath'] = intpath

# Constant verification hyperparameters for the ablation
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
ver_args['modelpath'] = modelpath

# Preallocate conformance confidences and safety chances
gamma = []
beta = []

# Loop through alphas; generate intervals; run validation; generate prism model; run verification
test_alphas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3]
for alpha in test_alphas:

    # Generate new intervals with this alpha
    print(f"Generating intervals with alpha = {alpha}...")
    int_args['alpha'] = alpha
    generate_intervals(int_args)

    ## Verification
    print(f"Running verification subprocess with alpha = {alpha}...")
    generate_prism_model(ver_args)
    term_cmd = [prismpath, "-cuddmaxmem", f"{cudd_mem}g", "-javamaxmem", f"{java_mem}g", modelpath, proppath]
    result = subprocess.run(term_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pmin = float(re.search(r"Result:\s*([0-9.]+)", result.stdout).group(1))
    beta.append(pmin)
    print("    >", re.search(r"^Time for model construction:.*$", result.stdout, re.MULTILINE).group(0))
    print("    >", f"Minimum safety chance: {pmin}")

    ## Validation
    print(f"Running validation with alpha = {alpha}...")
    confs = run_validation(val_args)
    median_conf = np.median(confs[0])
    gamma.append(median_conf)
    print("    >", f"Median confidence for alpha {alpha}: {median_conf}")
