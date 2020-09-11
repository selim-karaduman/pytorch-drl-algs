import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOPolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=0):
        super(PPOPolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)    
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
            nn.Linear(128, 1)
        )
      
    def forward(self, state):
        ac_probs =  self.actor(state)
        value = self.critic(state)
        dist = torch.distributions.Categorical(ac_probs)
        return dist, value


class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, H1=64, H2=64, seed=0):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)    
        self.network = nn.Sequential(
            nn.Linear(state_size, H1),
            nn.Tanh(),
            nn.Linear(H1, H2),
            nn.Tanh(),
            nn.Linear(H2, 1)
        )
        list(self.network)[-1].weight.data.mul_(0.1)
        list(self.network)[-1].bias.data.mul_(0.)

    def forward(self, state):
        value = self.network(state)
        return value


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, H1=64, H2=64, seed=0):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)    
        self.network = nn.Sequential(
            nn.Linear(state_size, H1),
            nn.Tanh(),
            nn.Linear(H1, H2),
            nn.Tanh(),
            nn.Linear(H2, action_size),
            nn.Softmax(dim=-1)
        )
        list(self.network)[-2].weight.data.mul_(0.1)
        list(self.network)[-2].bias.data.mul_(0.)

    def forward(self, state):
        ac_probs =  self.network(state)
        dist = torch.distributions.Categorical(ac_probs)
        return dist
