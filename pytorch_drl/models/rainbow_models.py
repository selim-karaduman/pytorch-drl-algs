import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_drl.utils.layers.noisy_linear import NoisyLinear

class RainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, atoms, seed=0):
        super(RainbowNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        self.atoms = atoms

        self.feature = nn.Linear(state_size, 128)
        
        self.advantage_n1 = NoisyLinear(128, 128)
        self.advantage_n2 = NoisyLinear(128, action_size*atoms)
        
        self.value_n1 = NoisyLinear(128, 128)
        self.value_n2 = NoisyLinear(128, 1*atoms)
        
        
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

    def __init__(self, state_size, action_size, H=128, seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)    
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

    def __init__(self, state_size, action_size, n_quants, seed=0):
        super(QRDQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        self.n_quants = n_quants

        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size*n_quants)
        )
        
        
    def forward(self, state):
        features = self.feature(state)
        features = features.view(-1, self.action_size, self.n_quants)
        return features


class QRRainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, atoms, seed=0):
        super(QRRainbowNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        self.atoms = atoms

        self.feature = nn.Linear(state_size, 128)
        
        self.advantage_n1 = NoisyLinear(128, 128)
        self.advantage_n2 = NoisyLinear(128, action_size*atoms)
        
        self.value_n1 = NoisyLinear(128, 128)
        self.value_n2 = NoisyLinear(128, 1*atoms)
        
        
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

    def __init__(self, state_size, action_size, seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)    
        
        self.linear = nn.Linear(state_size, 128)
        self.n2 = NoisyLinear(128, 128)
        self.n3 = NoisyLinear(128, action_size)
        
      
    def forward(self, state):
        x = F.relu(self.linear(state))
        x = F.relu(self.n2(x))
        return self.n3(x)

    def reset_noise(self):
        self.n2.reset_noise()
        self.n3.reset_noise()
        
class NoisyDuelingRainbowNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        

        self.feature = nn.Linear(state_size, 128)
        self.advantage_n1 = NoisyLinear(128, 128)
        self.advantage_n2 = NoisyLinear(128, action_size)
        self.value_n1 = NoisyLinear(128, 128)
        self.value_n2 = NoisyLinear(128, 1)
        
        
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

    def __init__(self, state_size, action_size, seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
            )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
            )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            )
        
    def forward(self, state):
        x = self.feature(state)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean())

