# Libraries
import numpy as np
import numpy as np
try: # fixes package incompatibility between numpy and gym
    np.bool8
except AttributeError:
    np.bool8 = np.bool_
import gym
import pickle as pkl
from scipy.stats import beta
from models import VisionNet
import torch
import cv2
import os
import sys

def generate_intervals_full(args):

    # Unpack arguments
    sx = args['sx']  # state space resolution
    x_space = args['x_space']  # state space range
    serr = args['serr']  # error space resolution
    err_space = args['err_space']  # error space range
    alpha = args['alpha']  # confidence level for confidence intervals
    savepath = args['savepath']  # path to the intervals file

    # Load network
    PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
    state_dict = torch.load(
        f"{PATH}/sample-weights/exampleStateEstimator.pth",
        weights_only=True,
    )
    network = VisionNet(
        num_conv_layers=2,
        input_dim=[400, 600],
        kernels=[32, 24],
        stride=2,
        conv_in_channels=[1, 16],
        conv_out_channels=[16, 16],
        pool_size=16,
        pool_stride=2,
        num_lin_layers=2,
        linear_layer_size=100,
        out_size=1,
    )
    network.load_state_dict(state_dict)
    network.eval()

    # Frame preprocessing (with optional noise)
    def preprocess(frame_rgb, noise_std):

        # Resize to training resolution (in case render settings differ)
        frame_rgb = cv2.resize(frame_rgb, (600, 400), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Add Gaussian noise before inversion so the statistics match training
        if noise_std > 0:
            noise = np.random.normal(loc=0.0, scale=noise_std, size=gray.shape).astype(np.float32)
            gray = np.clip(gray + noise, 0.0, 1.0)
        gray = 1.0 - gray

        # Convert to tensor (N,C,H,W) and push to device
        tensor = (
            torch.from_numpy(gray)
                .unsqueeze(0)      # channel dim
                .unsqueeze(0)      # batch dim
                .to(torch.float32)
        )
        return tensor

    # Define discrete space
    xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)
    errbar_space = np.arange(int(err_space[0]/serr), int(err_space[1]/serr))
    num_bins_err = len(errbar_space)
    edges = np.arange(err_space[0], err_space[1] + serr, serr, dtype=float)

    # Preallocate relevant data
    lower_bounds = []
    upper_bounds = []

    # Loop through all discrete positions
    SEED = 23
    env = gym.make("MountainCar-v0", render_mode="rgb_array").unwrapped
    alpha = alpha / (num_bins_err) # Bonferoni correction
    for xbar in xbar_space:

        # Collect data in this bin
        num_samples = 10
        x = []
        x_hat = []
        env.reset(seed=SEED)
        for i in range(num_samples):

            # Randomly select a pose
            p = np.random.uniform(xbar*sx, (xbar*sx)+sx)
            s = [p, 0]

            # Place mountaincar here; run estimation
            env.state = np.asarray(s, dtype=np.float32)
            frame = env.render()
            frame = preprocess(frame, noise_std=0.1000)
            with torch.no_grad():
                p_est = network(frame).cpu().item()
            
            # Allocate
            x.append(p)
            x_hat.append(p_est)
        
        # Convert to numpy; compute error
        x = np.array(x)
        x_hat = np.array(x_hat)
        errors = x - x_hat

        # Determine counts in each error bin
        counts, _ = np.histogram(errors, bins=edges)
        counts = counts + np.ones_like(counts) # condition to prevent nan uppers
        N = counts.sum()
        K = len(counts)

        # Construct intervals
        lowers = []
        uppers = []
        for ni in counts:
            lowers.append(beta.ppf(alpha / 2, ni, N - ni + 1))
            uppers.append(beta.ppf(1 - alpha / 2, ni + 1, N - ni))
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        # Append to external array
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
