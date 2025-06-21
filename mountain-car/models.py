# Libraries
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# Neural Net class
class VisionNet(nn.Module):
    def __init__(self, num_conv_layers = 3, \
                 input_dim = [400,600], \
                 kernels = [3,4,5], \
                 stride = 2, \
                 conv_in_channels= [3, 100, 100], \
                 conv_out_channels = [100, 100, 100], \
                 pool_size = 2, \
                 pool_stride = 2, \
                 num_lin_layers = 2, \
                 linear_layer_size = 100, \
                 out_size = 1):
        super().__init__()
        layer_list = []
        for i in range(num_conv_layers):
            layer_list.append(nn.Conv2d(in_channels=conv_in_channels[i], 
                                        out_channels=conv_out_channels[i], 
                                        kernel_size=kernels[i], stride=stride))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.MaxPool2d(pool_size, pool_stride))
        layer_list.append(nn.Flatten())
        
        self.feature_extractor = nn.Sequential(*layer_list)

        n_channels = self.feature_extractor(torch.empty(1, conv_in_channels[0], input_dim[0], input_dim[1])).size(-1)
        linear_list = []
        for i in range(num_lin_layers):
            if i == 0:
                linear_list.append( nn.Linear(n_channels, linear_layer_size))
            elif i<num_lin_layers-1:
                linear_list.append( nn.Linear(linear_layer_size, linear_layer_size))
            else:
                linear_list.append( nn.Linear(linear_layer_size, out_size))
            if i == num_lin_layers-1:
                linear_list.append(nn.Tanh())
            else:
                linear_list.append(nn.ReLU())
            
        self.classifier = nn.Sequential(*linear_list)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return 0.9*out-0.3 # scales and shifts the output to be only in the state space of MC
    
## Deep Q-network
class DQN(nn.Module):

    # Initialize the neural network
    def __init__(self, state_dim, action_dim, hidden_dim):
        '''
        INPUTS
        state_dim: dimension of the input state
        action_dim: dimension of the output action
        hidden_dim: dimension of the hidden layer
        '''
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # Input layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output layer
        )
    
    # Forward propagation of an observation "o"
    def forward(self, x):
        return self.net(x)

## Replay buffer to manage (s, a, r, s') experience
class ReplayBuffer:

    # Initialize double-ended queue collection to effcieintly manage experiences
    def __init__(self, buffer_size, batch_size):
        '''
        INPUTS
        buffer_size: the number of tuples stored in queue
        batch_size: the size of the minibatch randomly sampled from the buffer
        '''
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    # Add to queue
    def push(self, state, action, reward, next_state, done):
        '''
        INPUTS
        state: current system state
        action: action taken at this state
        reward: reward achieved by taking action at this state
        next_state: the state reached after takign the action
        done: boolean (whether the goal has been met)
        '''
        self.buffer.append((state, action, reward, next_state, done))

    # Randomly sample a mini-batch from the replay buffer
    def sample(self):
        '''
        OUTPUTS
        <tuple>: minibatch from replay buffer formatted as tensors
        '''
        batch = random.sample(self.buffer, self.batch_size) # rand pull from deque
        states, actions, rewards, next_states, dones = zip(*batch) # extract
        return (torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.BoolTensor(np.array(dones)))
