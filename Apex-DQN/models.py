import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ray
import random
from copy import deepcopy


def create_learner(network_type, num_inputs, num_actions, num_quantiles=4):
    if network_type == "DQN":
        return DQN(num_inputs, num_actions)
    elif network_type == "QuantileDQN":
        return QuantileDQN(num_inputs, num_actions, num_quantiles)
    elif network_type == "ConvDQN":
        return ConvDQN(num_inputs, num_actions)
    elif network_type == "ConvQuantileDQN":
        return ConvQuantileDQN(num_inputs, num_actions, num_quantiles)
    else:
        raise ValueError("network type not recognized. Options:"
                         " DQN / QunatileDQN / ConvDQN / ConvQuantileDQN")
        

class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(DQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs[0], hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class QuantileDQN(nn.Module):

    def __init__(self, num_inputs, num_actions, num_quantiles,\
                 hidden_size=256, init_w=3e-3):
        super(QuantileDQN, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.linear1 = nn.Linear(num_inputs[0], hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_quantiles * num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.tau = (torch.FloatTensor((2 * np.arange(num_quantiles) + 1) /
                     (2.0 * num_quantiles)).view(1, -1))

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x.view(-1, self.num_actions, self.num_quantiles)
        

class ConvDQN(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(ConvDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_input_dim = self.feature_size()
        self.linear1 = nn.Linear(self.fc_input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        
        x = F.relu(self.linear1(features))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.num_inputs)).view(1, -1).size(1)


class ConvQuantileDQN(nn.Module):

    def __init__(self, num_inputs, num_actions, num_quantiles,\
                 hidden_size=256, init_w=3e-3):
        super(ConvQuantileDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_input_dim = self.feature_size()
        self.linear1 = nn.Linear(self.fc_input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_quantiles * num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.tau = (torch.FloatTensor((2 * np.arange(num_quantiles) + 1) /
                     (2.0 * num_quantiles)).view(1, -1))

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)

        x = F.relu(self.linear1(features))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x.view(-1, self.num_actions, self.num_quantiles)
    
    def feature_size(self):
        return self.conv(torch.zeros(1, *self.num_inputs)).view(1, -1).size(1)