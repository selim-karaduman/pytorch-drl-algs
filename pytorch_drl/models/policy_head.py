"""
From:

https://github.com/ikostrikov
/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py

https://github.com/ikostrikov
/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py

MIT License

Copyright (c) 2017 Ilya Kostrikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli
import gym

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample()

    def log_prob(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1, keepdims=True)   
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdims=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions)\
                .view(actions.size(0), -1).sum(-1, keepdims=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    
    def log_prob(self, actions):
        return super().log_prob(actions)\
                .view(actions.size(0), -1).sum(-1, keepdims=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


# ======================================================================

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class CategoricalHead(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CategoricalHead, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussianHead(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussianHead, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        #self.logstd = AddBias(torch.zeros(num_outputs))
        self.fc_logstd = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        """
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        """
        action_logstd = self.fc_logstd(x)
        return FixedNormal(action_mean, action_logstd.exp())


class BernoulliHead(nn.Module):
    # NOT TESTED
    def __init__(self, num_inputs, num_outputs):
        super(BernoulliHead, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

# ======================================================================


class PolicyHead(nn.Module):
    def __init__(self, num_inputs, num_outputs, action_space):
        super(PolicyHead, self).__init__()
        self.action_space = action_space
        if isinstance(self.action_space, gym.spaces.Box):
            self.distribution = DiagGaussianHead(num_inputs, num_outputs)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.distribution = CategoricalHead(num_inputs, num_outputs)
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            self.distribution = BernoulliHead(num_inputs, num_outputs)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            self.distribution = MultiCategoricalHead(num_inputs, num_outputs)
        else:
            print("Only Box, Discrete, Multi-Binary, Multi-Discete\
                     spaces can be used")
            raise ValueError

    def forward(self, x):
        return self.distribution(x)
