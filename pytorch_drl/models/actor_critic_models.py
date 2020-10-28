import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_drl.models.policy_head import PolicyHead

class ActorCriticMLP(nn.Module):

    def __init__(self, state_size, action_size, action_space, H=128):
        super(ActorCriticMLP, self).__init__()    
        self.actor = nn.Sequential(
            nn.Linear(state_size, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            PolicyHead(H, action_size, action_space)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 1)
        )
      
    def forward(self, state):
        dist =  self.actor(state)
        value = self.critic(state)
        return dist, value



class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, H1=128, H2=128):
        super(CriticNetwork, self).__init__()   
        self.network = nn.Sequential(
            nn.Linear(state_size, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, 1)
        )
        list(self.network)[-1].weight.data.mul_(0.1)
        list(self.network)[-1].bias.data.mul_(0.)

    def forward(self, state):
        value = self.network(state)
        return value


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, action_space, 
                    H1=128, H2=128):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            PolicyHead(H2, action_size, action_space)
        )
        
    def forward(self, state):
        dist =  self.network(state)
        return dist
