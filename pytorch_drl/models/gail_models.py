import torch
import torch.nn as nn
import torch.nn.functional as F

class GAILDiscriminator(nn.Module):

    def __init__(self, state_size, action_size):
        super(GAILDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
