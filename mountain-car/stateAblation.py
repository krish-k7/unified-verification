# Libraries
import numpy as np
from prismgen import generate_prism_model
from genIntervals import generate_intervals_full
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
proppath = f"{PATH}/prism-models/mc-props.props" # path to the MC PCTL properties for any prism model
prismpath = f"{SUPER_PATH}/prism_ws/bin/prism"

# Reusable parameters
sx = 0.05  # state space resolution
x_space = [-1.2, 0.6]  # state space range
sv = 0.01 # velocity space resolution
v_space = [-0.07, 0.07] # velocity space range
serr = 0.1 # error space resolution
err_space = [-0.2, 0.4] # error space range
cudd_mem = 4 # memory for CUDD in GB
java_mem = 4 # memory for Java in GB

# Constant interval generation hyperparameters for the ablation
int_args = {}
int_args['sx'] = sx
int_args['x_space'] = x_space
int_args['serr'] = serr
int_args['err_space'] = err_space
int_args['savepath'] = intpath
int_args['alpha'] = 0.1
print(f"Generating intervals with constant alpha = 0.1...")
generate_intervals_full(int_args)

# Initial states to test
initial_states = [
    [0.35, 0.07],
    [0.40, 0.06],
]

# Preallocate safety chances
beta = []

# Loop through initial states; generate prism model; run verification
for state in initial_states:
    init_state = [int(np.floor(state[0] / sx)), int(np.floor(state[1] / sv))] # abstract initial state

    # Verification arguments
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
    