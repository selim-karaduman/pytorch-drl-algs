import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def init_fan_in(layer):
    fan_in = layer.shape[0]
    rang = 1. / np.sqrt(fan_in)
    layer.data.uniform_(-rang, rang)

def init_uniform(layer, l=-3e-3, r=3e-3):
    layer.data.uniform_(l, r)
          
class DDPGValueNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed=0, H1=64, H2=64):
        super(DDPGValueNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, H1)
        self.fc2 = nn.Linear(H1+action_size, H2)
        self.fc3 = nn.Linear(H2, 1)

        init_fan_in(self.fc1.weight.data)
        init_fan_in(self.fc2.weight.data)
        init_uniform(self.fc3.weight.data)

    def forward(self, state, action):
        feats = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((feats, action), dim=1)))
        x = self.fc3(x)
        return x

class DDPGPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, H1=64, H2=64):
        super(DDPGPolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, 1)

        init_fan_in(self.fc1.weight.data)
        init_fan_in(self.fc2.weight.data)
        init_uniform(self.fc3.weight.data)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
