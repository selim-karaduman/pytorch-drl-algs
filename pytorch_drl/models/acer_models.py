import torch
import torch.nn as nn
import torch.nn.functional as F

class ACERModel(nn.Module):

    def __init__(self, state_size, action_size):
        super(ACERModel, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size)
        )
      
    def forward(self, state):
        policy =  self.actor(state)
        value = self.critic(state)
        return policy, value