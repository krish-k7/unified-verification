# Libraries
from models import DQN, ReplayBuffer as RB
import numpy as np
try: # fixes package incompatibility between numpy and gym
    np.bool8
except AttributeError:
    np.bool8 = np.bool_
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import random
import pickle as pkl
import os
import sys

# Push to CUDA device
if torch.cuda.is_available():
    print("A CUDA device was found and will be used")
    device = torch.device("cuda")
else:
    print("A CUDA device was not found, so the CPU will be used instead")
    device = torch.device("cpu")

# Hyperparameters
GAMMA = 0.99 # discount factor
LR = 0.001 # optimizer lr
BUFFER_SIZE = 10000 # replay memory
BATCH_SIZE = 64 # replay mini-batch size
EPSILON = 1.0 # epislon-greedy
MIN_EPSILON = 0.0
TARGET_UPDATE = 50 # number of episodes used to update the target network
EPISODE_LENGTH = 1000 # number of time steps in each episode
TRAIN_EPISODES = 5000 # number of training episodes
HIDDEN_DIM = 128 # dimension of hidden layer
PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path

# Make the gym environment
base_env = gym.make('MountainCar-v0')
state_dim = base_env.observation_space.shape[0]
action_dim = base_env.action_space.n

# Wrap environment in noisy observation space
class NoisyWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma=0.1):
        super().__init__(env)
        self.sigma = sigma

    def observation(self, obs):
        return obs + np.random.normal(0, self.sigma, size=obs.shape)
env = NoisyWrapper(base_env)

# Instantiate policy and target networks (same architecture) and replay buffer
policy_net = DQN(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(device)
target_net = DQN(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
buffer = RB(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

# Instantiate optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# Training loop for DQN
episode_rewards = []
episode_losses = []
epsilon = EPSILON
epsilon_decay = 0.9983 # MIN_EPSILON**(1/TRAIN_EPISODES)
for episode in range(TRAIN_EPISODES):
    
    # Step through single episode
    state, _ = env.reset()
    total_reward = 0.0
    losses_per_episode = []
    for i in range(EPISODE_LENGTH):

        # Choose action according to epsilon-greedy policy
        if random.random() < epsilon: # exploration
            action = env.action_space.sample()
        else: # exploitation
            with torch.no_grad(): # no gradient computation - for efficiency
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_t)
                action = q_values.argmax(dim=1).item()
        
        # Propagate state and extract reqard; store in replay buffer
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Learn once there is enough experience stored in buffer
        if len(buffer.buffer) >= BATCH_SIZE:

            # Sample from buffer and compute q-values
            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample()

            # Push to device
            states_b = states_b.to(device)
            actions_b = actions_b.to(device)
            rewards_b = rewards_b.to(device)
            next_states_b = next_states_b.to(device)
            dones_b = dones_b.to(device)

            # Compute Q-value estimates
            q_values = policy_net(states_b)
            q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)
            
            # Compute q-values from target network
            with torch.no_grad():
                next_q_values = target_net(next_states_b).max(dim=1)[0]
                next_q_values[dones_b] = 0.0
            
            # Compute target and MSE loss
            target = rewards_b + GAMMA * next_q_values
            loss = nn.MSELoss()(q_values, target)
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store loss for analysis
            losses_per_episode.append(loss.item())
        
        # Break loop if done
        if done:
            break
    
    # Update epsilon
    epsilon = max(MIN_EPSILON, epsilon*epsilon_decay)
    
    # Update target network
    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # Log
    episode_rewards.append(total_reward)
    if len(losses_per_episode) > 0:
        episode_losses.append(np.mean(losses_per_episode))
    else:
        episode_losses.append(0.0)
    
    # Print progress
    if (episode+1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode: {episode+1}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

env.close() # close gym environment

# Save policy and training params/results
torch.save(policy_net.state_dict(), f"{PATH}/sample-weights/policy.pth") # or move up to save checkpoints
training_params = {
    "gamma":GAMMA,
    "learning_rate":LR,
    "buffer_size":BUFFER_SIZE,
    "batch_size":BATCH_SIZE,
    "epsilon":EPSILON,
    "min_epsilon":MIN_EPSILON,
    "decay_rate":epsilon_decay,
    "target_update":TARGET_UPDATE,
    "episode_length":EPISODE_LENGTH,
    "train_episodes":TRAIN_EPISODES,
    "hidden_dim":HIDDEN_DIM}
training_metrics = {
    "rewards": episode_rewards,
    "losses": episode_losses}
with open(f"{PATH}/training_params.pkl", "wb") as f:
    pkl.dump(training_params, f)
with open(f"{PATH}/training_metrics.pkl", "wb") as f:
    pkl.dump(training_metrics, f)
