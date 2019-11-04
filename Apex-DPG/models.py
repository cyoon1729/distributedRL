import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ray
import random
from copy import deepcopy


def create_critic(network_type, num_inputs, num_actions, num_quantiles=4):
    if network_type == "DQN":
        return DQN(num_inputs[0], num_actions[0], num_quantiles)
    else:
        raise ValueError("network type not recognized. Options:"
                         "DQN")
        

def create_policy(num_inputs, num_actions):
    return DeterministicPolicy(num_inputs[0], num_actions[0])
    

class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions, num_quantiles,\
                 hidden_size=256, init_w=3e-3):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DeterministicPolicy(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x