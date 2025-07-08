# Libraries
import numpy as np
from genIntervals import generate_intervals_full
from getData import get_data, get_data_comp
from validate import run_validation
from prismgen import generate_prism_model
from pathlib import Path
import sys
import os
import subprocess
import re
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # path to the file folder
STUDY_PATH = f"{PATH}/data/alpha-ablation-data"
HOME = Path.home()

# Reusable paths
trainpath = f"{STUDY_PATH}/train-data.pkl" # path to save the training data
testpath = f"{STUDY_PATH}/test-data.pkl" # path to save the test data
intpath = f"{STUDY_PATH}/tmp-intervals.pkl" # path to save the intervals file
modelpath = f"{STUDY_PATH}/imdp.pm" # path to save the prism model
proppath = f"{PATH}/prism-models/mc-props.props" # path to the MC PCTL properties for any prism model
prismpath = f"{HOME}/prism_ws/bin/prism"

# Reusable parameters
sx = 0.05  # state space resolution
x_space = [-1.2, 0.6]  # state space range
sv = 0.01 # velocity space resolution
v_space = [-0.07, 0.07] # velocity space range
serr = 0.1 # error space resolution
err_space = [-0.2, 0.4] # error space range
cudd_mem = 4 # memory for CUDD in GB
java_mem = 4 # memory for Java in GB

# Initial mountain car state
init_state = [0.35, 0.07]
init_state = [int(np.floor(init_state[0] / sx)), int(np.floor(init_state[1] / sv))] # abstract initial state

# Generate training and validation data
data_args = {}
data_args['sx'] = sx
data_args['x_space'] = x_space
data_args['num_samps'] = 10 # number of trials to run
data_args['noise_std'] = 0.1 # noise standard deviation
# data_args['savepath'] = trainpath
# get_data_comp(data_args) # generate training data
data_args['savepath'] = testpath
# print("Collecting test trajectories...\n")
# get_data(data_args) # generate test data
# print("\n")

# Constant interval generation hyperparameters for the ablation
int_args = {}
int_args['sx'] = sx
int_args['x_space'] = x_space
int_args['serr'] = serr
int_args['err_space'] = err_space
int_args['datapath'] = trainpath
int_args['savepath'] = intpath

# Constant validation hyperparameters for the ablation
val_args = {}
val_args['sx'] = sx
val_args['x_space'] = x_space
val_args['serr'] = serr
val_args['err_space'] = err_space
val_args['num_instances'] = 1 # number of validation instances
val_args['testpath'] = testpath
val_args['intpath'] = intpath

# Constant verification hyperparameters for the ablation
ver_args = {}
ver_args['sx'] = sx
ver_args['x_space'] = x_space
ver_args['sv'] = sv
ver_args['v_space'] = v_space
ver_args['serr'] = serr
ver_args['err_space'] = err_space
ver_args['init_state'] = init_state
ver_args['intpath'] = intpath
ver_args['modelpath'] = modelpath

# Preallocate conformance confidences and safety chances
gamma = []
beta = []

# Loop through alphas; generate intervals; run validation; generate prism model; run verification
test_alphas = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3]
for alpha in test_alphas:

    # Generate new intervals with this alpha
    print(f"Generating intervals with alpha = {alpha}...")
    int_args['alpha'] = alpha
    generate_intervals_full(int_args)

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

# Plot beta vs test_alphas
plt.figure(figsize=(10, 5))
plt.plot(test_alphas, beta, marker='o', label='Safety Chances (Beta)')
plt.xlabel('Alpha')
plt.ylabel('1 - Beta')
plt.title('Safety Chances vs Alpha')
plt.grid(True)
plt.legend()
plt.show()

# Plot gamma vs test_alphas
plt.figure(figsize=(10, 5))
plt.plot(test_alphas, gamma, marker='o', label='Conformance Confidence (Gamma)', color='orange')
plt.xlabel('Alpha')
plt.ylabel('1 - Gamma')
plt.title('Conformance Confidence vs Alpha')
plt.grid(True)
plt.legend()
plt.show()

