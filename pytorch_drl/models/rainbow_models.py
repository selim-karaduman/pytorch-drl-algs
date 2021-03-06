import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_drl.utils.layers.noisy_linear import NoisyLinear

class RainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, atoms, H=64):
        super(RainbowNetwork, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.atoms = atoms

        self.feature = nn.Linear(state_size, H)
        
        self.advantage_n1 = NoisyLinear(H, H)
        self.advantage_n2 = NoisyLinear(H, action_size*atoms)
        
        self.value_n1 = NoisyLinear(H, H)
        self.value_n2 = NoisyLinear(H, 1*atoms)
        
        
    def forward(self, state, log=False):
        features = F.relu(self.feature(state))
        
        advantage = F.relu(self.advantage_n1(features))
        advantage = self.advantage_n2(advantage)
        
        value = F.relu(self.value_n1(features))
        value = self.value_n2(value)
        
        advantage = advantage.view(-1, self.action_size, self.atoms)
        value = value.view(-1, 1, self.atoms)
        q = value + (advantage - advantage.mean(1, keepdim=True))
        
        if log:
            q = F.log_softmax(q, dim=2)
        else:
            q = F.softmax(q, dim=2)
        return q

    def reset_noise(self):
        self.advantage_n1.reset_noise()
        self.advantage_n2.reset_noise()
        self.value_n1.reset_noise()
        self.value_n2.reset_noise()




class DQNNetwork(nn.Module):

    def __init__(self, state_size, action_size, H=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, action_size)
        )
      
    def forward(self, state):
        return self.feature(state)
        

class QRDQNNetwork(nn.Module):

    def __init__(self, state_size, action_size, n_quants, H=64):
        super(QRDQNNetwork, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.n_quants = n_quants

        self.feature = nn.Sequential(
            nn.Linear(state_size, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, action_size*n_quants)
        )
        
        
    def forward(self, state):
        features = self.feature(state)
        features = features.view(-1, self.action_size, self.n_quants)
        return features


class QRRainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, atoms, H=64):
        super(QRRainbowNetwork, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.atoms = atoms

        self.feature = nn.Linear(state_size, H)
        
        self.advantage_n1 = NoisyLinear(H, H)
        self.advantage_n2 = NoisyLinear(H, action_size*atoms)
        
        self.value_n1 = NoisyLinear(H, H)
        self.value_n2 = NoisyLinear(H, 1*atoms)
        
        
    def forward(self, state, log=False):
        features = F.relu(self.feature(state))
        
        advantage = F.relu(self.advantage_n1(features))
        advantage = self.advantage_n2(advantage)
        
        value = F.relu(self.value_n1(features))
        value = self.value_n2(value)
        
        advantage = advantage.view(-1, self.action_size, self.atoms)
        value = value.view(-1, 1, self.atoms)
        q = value + (advantage - advantage.mean(1, keepdim=True))
        return q

    def reset_noise(self):
        self.advantage_n1.reset_noise()
        self.advantage_n2.reset_noise()
        self.value_n1.reset_noise()
        self.value_n2.reset_noise()


#-----------------------------------------------------------------------------

# For testing specific algs:


class NoisyRainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, H=64):
        super().__init__()        
        self.linear = nn.Linear(state_size, H)
        self.n2 = NoisyLinear(H, H)
        self.n3 = NoisyLinear(H, action_size)
        
      
    def forward(self, state):
        x = F.relu(self.linear(state))
        x = F.relu(self.n2(x))
        return self.n3(x)

    def reset_noise(self):
        self.n2.reset_noise()
        self.n3.reset_noise()
        
class NoisyDuelingRainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, H=64):
        super().__init__()
        self.feature = nn.Linear(state_size, H)
        self.advantage_n1 = NoisyLinear(H, H)
        self.advantage_n2 = NoisyLinear(H, action_size)
        self.value_n1 = NoisyLinear(H, H)
        self.value_n2 = NoisyLinear(H, 1)
        
        
    def forward(self, state, log=False):
        features = F.relu(self.feature(state))
        
        advantage = F.relu(self.advantage_n1(features))
        advantage = self.advantage_n2(advantage)
        
        value = F.relu(self.value_n1(features))
        value = self.value_n2(value)

        q = value + (advantage - advantage.mean())
        return q

    def reset_noise(self):
        self.advantage_n1.reset_noise()
        self.advantage_n2.reset_noise()
        self.value_n1.reset_noise()
        self.value_n2.reset_noise()


class DuelingRainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, H=64):
        super().__init__()        
        self.feature = nn.Sequential(
            nn.Linear(state_size, H),
            nn.ReLU()
            )

        self.advantage = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, action_size)
            )

        self.value = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 1)
            )
        
    def forward(self, state):
        x = self.feature(state)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean())

