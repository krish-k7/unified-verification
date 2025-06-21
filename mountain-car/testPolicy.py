# Libraries
from models import DQN
import numpy as np
try: # again, fixes package incompatibility between numpy and gym
    np.bool8
except AttributeError:
    np.bool8 = np.bool_
import gym
import torch
import pickle as pkl
import matplotlib.pyplot as plt
import time
import random
import os
import sys

VERSION = 2

# Create MC environment
env = gym.make('MountainCar-v0', render_mode="human")
# env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load in the trained policy
PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
state_dict = torch.load(
    f"{PATH}/sample-weights/policy_{VERSION}.pth",
    map_location=torch.device('cpu')
)
policy_net = DQN(state_dim, action_dim, 128)
policy_net.load_state_dict(state_dict)
policy_net.eval()

# Deploy the policy
state, _ = env.reset()
done = False
time = 0
while not done:

    time += 1
    # print(f"Time = {time}")

    # Add noise to state
    cov = [[0.0, 0], [0, 0.0]]
    noisy_state = np.random.multivariate_normal(state, cov, size=1)[0]
    
    # Compute action from learned policy
    with torch.no_grad():
        state_t = torch.FloatTensor(noisy_state).unsqueeze(0)
        q_values = policy_net(state_t)
        action = q_values.argmax(dim=1).item()

    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()