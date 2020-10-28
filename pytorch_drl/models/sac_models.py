import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SACPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, H1=64, H2=64, 
                min_log_std=-20, max_log_std=2):
        super(SACPolicyNetwork, self).__init__()
        self.eps = 1e-8
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.fc1 = nn.Linear(state_size, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3_mu = nn.Linear(H2, action_size)
        self.fc3_log_std = nn.Linear(H2, action_size)

    def forward(self, state, noise=True):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        log_std = self.fc3_log_std(x).clamp_(self.min_log_std, self.max_log_std)
        std = log_std.exp()

        if not noise:
            action = torch.tanh(mu)
            log_prob = None
        else:
            dist = Normal(mu, std)
            u = dist.rsample()
            action = torch.tanh(u) # squashed action
            log_det_tanh = (2*(np.log(2) - u - F.softplus(-2*u))).sum(1, keepdim=True)
            log_prob = dist.log_prob(u).sum(1, keepdim=True) - log_det_tanh
        return action, log_prob

class SACValueNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, H1=64, H2=64):
        super(SACValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
