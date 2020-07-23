import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
# From: https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb

class NoisyLinear(nn.Module):

    def __init__(self, in_size, out_size, std_init=0.4):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.std_init = std_init

        self.weight_mu = Parameter(torch.Tensor(out_size, in_size))
        self.weight_sigma = Parameter(torch.Tensor(out_size, in_size))
        self.register_buffer('weight_eps', torch.empty(out_size, in_size))

        self.bias_mu = Parameter(torch.Tensor(out_size))
        self.bias_sigma = Parameter(torch.Tensor(out_size))
        self.register_buffer('bias_eps', torch.empty(out_size))
        
        self.reset_parameters()
        self.reset_noise()

    def _scale_noise(self, size):
        t = torch.randn(size)
        return t.sign().mul_(t.abs().sqrt_())

    def reset_noise(self):
        # Reset epsilon buffers: sample in_size+out_size parameters
        #   Then, take outer product to get the epsilon
        v_in = self._scale_noise(self.in_size)
        v_out = self._scale_noise(self.out_size)
        self.weight_eps.copy_(v_out.ger(v_in))
        self.bias_eps.copy_(torch.randn(self.out_size))
        # self.bias_eps.copy_(v_out)
        

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_size)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_size))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_size))

    def forward(self, state):
        if self.training:
            return F.linear(state, 
                    self.weight_eps * self.weight_sigma + self.weight_mu,
                    self.bias_eps * self.bias_sigma + self.bias_mu)
        else:
            return F.linear(state, self.weight_mu, self.bias_mu)

