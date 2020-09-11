import torch
import torch.nn as nn
import torch.nn.functional as F

class A2CNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=0):
        super(A2CNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)    
        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
      
    def forward(self, state):
        ac_probs =  self.actor(state)
        value = self.critic(state)
        dist = torch.distributions.Categorical(ac_probs)
        return dist, value
