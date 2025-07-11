# Libraries
import numpy as np
try: # fixes package incompatibility between numpy and gym
    np.bool8
except AttributeError:
    np.bool8 = np.bool_
import gym
import cv2
import torch
from models import VisionNet as Net, DQN
import random
import os
import sys

# Wrapper environment for controlling MountainCar in a closed-loop with vision-based state estimation
class clsys():

    def __init__(self, sigma, seed):

        # Load vision-based estimator
        PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
        state_dict = torch.load(
            f"{PATH}/sample-weights/exampleStateEstimator.pth",
            weights_only=True)
        self.vision_net = Net(
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
            out_size=1)
        self.vision_net.load_state_dict(state_dict)
        self.vision_net.eval()

        # Load DQN controller
        self.policy_net = DQN(2, 3, 128)  # mountain car state dim: 2, mountain car action dim: 3
        self.policy_net.load_state_dict(
            torch.load(f"{PATH}/sample-weights/policy.pth", map_location=torch.device('cpu')))
        self.policy_net.eval()

        # Other items
        self.start_environment()
        self.state, _ = self.env.reset(seed=seed)
        self.time = 0
        self.last_pose_est = 0.0
        self.done = False
        self.sigma = sigma
        
        # Set RNG seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed

    # Propagate the system
    def step(self):

        # Render image; preprocess; make state estimation
        self.time += 1
        frame = self.env.render()
        x = self.preprocess(frame, noise_std=self.sigma)
        with torch.no_grad():
            pose_est = self.vision_net(x).cpu().item()
        if self.time == 1:
            self.last_pose_est = pose_est
        vel_est = np.clip(pose_est-self.last_pose_est, -0.07, 0.07)
        self.last_pose_est = pose_est

        # Compute control action from policy
        state_est = [pose_est, vel_est] # full observed
        with torch.no_grad():
            state_est = torch.FloatTensor(state_est).unsqueeze(0)#.to(device)
            q_values = self.policy_net(state_est)
            action = q_values.argmax(dim=1).item()
        
        # Propagate
        self.state, _, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated

        # Termination condition of env
        if self.done:
            self.close_environment()

        return self.state, state_est.tolist()[0]

    # Make new environment
    def start_environment(self):
        self.env = gym.make("MountainCar-v0", render_mode="rgb_array")

    # Close existing environment
    def close_environment(self):
        self.env.close()
    
    def reset(self):
        self.start_environment()
        self.env.reset(seed=self.seed)
        self.time = 0
        self.done = False
    
    def pseudostep(self, state, state_est):
        mc = self.env.unwrapped
        mc.state = state
        
    # Frame preprocessing (with optional noise)
    @staticmethod
    def preprocess(frame_rgb, noise_std):

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
        )
        return tensor
    

# Wrapper environment for stepping MountainCar for verification purposes
class clsysPseudo:
    def __init__(self, seed=0):
        PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
        self.rng = np.random.default_rng(seed)
        self.env = gym.make("MountainCar-v0").unwrapped
        self.env.reset(seed=seed)
        state_dict = torch.load(
            f"{PATH}/sample-weights/policy.pth",
            map_location=torch.device('cpu'))
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy_net = DQN(state_dim, action_dim, 128)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()

    def pseudostep(self, state_true, state_err):
        
        state_est = np.array([state_true[0]+state_err, state_true[1]], dtype=np.float32)

        # Compute control action
        with torch.no_grad():
            q = self.policy_net(torch.tensor(state_est).unsqueeze(0))
            action = int(q.argmax(1).item())

        # Set the real env to the desired true state and propagate
        self.env.state = np.asarray(state_true, dtype=np.float32)
        next_state, _, terminated, truncated, _ = self.env.step(action)
        return np.array(next_state), action, terminated or truncated

    def close(self):
        self.env.close()

# Sample usage
if __name__ == "__main__":
    sys = clsys(sigma=0.1, seed=23)
    for t in range(200):
        state, state_est = sys.step()
        state = [round(float(s), 4) for s in state]
        state_est = [round(float(s), 4) for s in state_est]
        print(f"Time step: {t}, State: {state}, Estimated State: {state_est}")
        if sys.done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    sys.close_environment()
