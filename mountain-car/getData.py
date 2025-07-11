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
from models import VisionNet, DQN
import pickle as pkl
import os
import sys

def get_data(args):

    # Unpack arguments
    num_samps = args['num_samps']  # number of trials to run
    noise_std = args['noise_std']  # noise standard deviation
    savepath = args['savepath']  # path to save the data
    display = args.get('display', False)  # whether to display the environment

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

    # Load DQN model
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dict = torch.load(
        f"{PATH}/sample-weights/policy.pth",
        map_location=torch.device('cpu')
    )
    policy_net = DQN(state_dim, action_dim, 128)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    # Loop through data collection instances
    state_data_train = []
    estimate_data_train = []
    num_success = 0
    num_steps = []
    seeds = np.arange(0, num_samps, 1) * 10
    for k in range(num_samps):
        if display:
            print(f"Executing trial {k}")

        # Deploy the policy and estimator
        state, _ = env.reset(seed=int(seeds[k]))
        done = False
        time = 0
        states = []
        estimates = []
        while not done:

            # Render image; preprocess; make state estimation
            time += 1
            frame = env.render()
            x = preprocess(frame, noise_std=noise_std)
            with torch.no_grad():
                pose_est = network(x).cpu().item()

            # Compute control action from estimated state
            state_est = [pose_est, state[1]] # partially observed
            with torch.no_grad():
                state_est = torch.FloatTensor(state_est).unsqueeze(0)#.to(device)
                q_values = policy_net(state_est)
                action = q_values.argmax(dim=1).item()

            # Record estimated/actual pose
            states.append(state)
            estimates.append(state_est.squeeze(0).cpu().numpy())
            
            # Propagate
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        state_data_train.extend(states)
        estimate_data_train.extend(estimates)
        num_steps.append(time)
        
        if display:
            print(f"Trial {k + 1}/{num_samps} completed.")
            if time >= 200:
                print("    > Result: Failed to reach the goal.")
            else:
                print("    > Result: Success! Reached the goal.")
                num_success += 1
            print(f"    > Time steps taken: {time}")
            print(f"    > Current success rate: {round(num_success / (k + 1), 4)}")

    env.close()

    # Final stats
    if display:
        print("Final Statistics:")
        print(f"    > Total trials: {num_samps}")
        print(f"    > Success rate: {num_success / num_samps:.2%}")
        print(f"    > Average time steps per trial: {np.mean(num_steps):.2f}")

    # Save output
    state_data_train = np.array(state_data_train)
    estimate_data_train = np.array(estimate_data_train)
    DATA = {
        "state_data_train": state_data_train,
        "estimate_data_train": estimate_data_train
    }
    with open(savepath, "wb") as f:
        pkl.dump(DATA, f)

# Sample usage
if __name__ == "__main__":

    # Find path to save test data file
    PATH = os.path.dirname(os.path.abspath(sys.argv[0]))

    args = {
        'sx': 0.05,
        'x_space': [-1.2, 0.6],
        'num_samps': 100, # really, num trajectories
        'noise_std': 0.01,
        'savepath': f'{PATH}/data/sample-test-data.pkl',
        'display': True, # Explicitly set to True to display the statistics
    }
    print("Collecting test trajectories...\n")
    get_data(args) # Call the function to collect data
    print("Data collection complete. Data saved to:", args['savepath'])
