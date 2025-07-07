# Libraries
import numpy as np
try: # fixes package incompatibility between numpy and gym
    np.bool8
except AttributeError:
    np.bool8 = np.bool_
# import gym
import gymnasium as gym
import cv2
import torch
from models import VisionNet as Net, DQN
import random
import os
import sys

# Set RNG seeds
SEED = 23
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Load network
PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
state_dict = torch.load(
    f"{PATH}/sample-weights/exampleStateEstimator.pth",
    weights_only=True,
)
network = Net(
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
def preprocess(frame_rgb, noise_std=0.00):

    # Resize to training resolution (in case render settings differ)
    frame_rgb = cv2.resize(frame_rgb, (600, 400), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Add Gaussian noise *before* inversion so the statistics match training
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
             #.to(device)
    )
    return tensor

# Load DQN model
env = gym.make("MountainCar-v0", render_mode="rgb_array")
# env = gym.make("MountainCar-v0", render_mode="human")
env.action_space.seed(SEED)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_dict = torch.load(
    f"{PATH}/sample-weights/policy.pth",
    map_location=torch.device('cpu')
)
policy_net = DQN(state_dim, action_dim, 128)
policy_net.load_state_dict(state_dict)
policy_net.eval()

# Deploy the policy and estimator
state, _ = env.reset(seed=SEED)
done = False
time = 0
states = []
state_estimates = []
last_pose_est = 0.0
while not done:

    # Render image; preprocess; make state estimation
    time += 1
    frame = env.render()
    x = preprocess(frame, noise_std=0.05)
    with torch.no_grad():
        pose_est = network(x).cpu().item()
    if time == 1:
        last_pose_est = pose_est
    vel_est = np.clip(pose_est-last_pose_est, -0.07, 0.07)
    last_pose_est = pose_est

    # Compute control action from estimated state
    state_est = [pose_est, state[1]] # partially observed
    with torch.no_grad():
        state_est = torch.FloatTensor(state_est).unsqueeze(0)#.to(device)
        q_values = policy_net(state_est)
        action = q_values.argmax(dim=1).item()

    print(f"step {time:03d} | predicted pose {pose_est:+.3f} | actual pose: {state[0]:+.3f} | predicted velo {vel_est:+.3f} | actual velo {state[1]:+.3f}")

    # Record estimated/actual pose
    state_estimates.append(state_est)
    states.append(state)
    
    # Propagate
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
if time >= 200:
    print("Failed!")
else:
    print("Success!")
env.close()
