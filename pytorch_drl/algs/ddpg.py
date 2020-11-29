import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
import random
import copy
from collections import deque
from pytorch_drl.utils.memory.buffer import PriorityBuffer, UniformBuffer
import pytorch_drl.utils.misc as misc
import pytorch_drl.utils.model_utils as model_utils
from pytorch_drl.algs.base import ValueBased

class DDPG(ValueBased):

    def __init__(self, 
                 policy_net=None,
                 value_net=None,
                 gamma=0.99, 
                 lr_val=1e-3,
                 lr_pol=1e-3,
                 buf_size=int(1e5),
                 batch_size=64,
                 tau=1e-3,
                 device="cpu",
                 max_grad_norm=0.5,
                 noise_process=None,
                 min_act=None,
                 max_act=None,
                 learn_every=1,
                 warm_up=1e2,
                 seed=0):
        super().__init__()
        self.policy_net = policy_net
        self.policy_net_target = copy.deepcopy(policy_net)
        self.value_net = value_net
        self.value_net_target = copy.deepcopy(value_net)
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.min_act = min_act
        self.max_act = max_act
        self.tau = tau
        self.learn_every = learn_every
        self.max_grad_norm = max_grad_norm
        self.seed = random.seed(seed)
        self.warm_up = warm_up
        self.noise_process = noise_process
        self.exp_index = 0
        self.value_net.to(device)        
        self.policy_net.to(device)
        self.value_net_target.to(device)        
        self.policy_net_target.to(device)
        self.val_optimizer = torch.optim.Adam(value_net.parameters(), 
                                                lr=lr_val)
        self.pol_optimizer = torch.optim.Adam(policy_net.parameters(),
                                                lr=lr_pol)
        self.replay_buffer = UniformBuffer(buf_size, batch_size, seed, 
                                            device, action_type=torch.float)

    def act(self, state, test=False):
        if len(self.replay_buffer) < self.warm_up: 
            return np.random.uniform(self.min_act, self.max_act)
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.policy_net.eval()
            action = self.policy_net(state)
            self.policy_net.train()
        action = action.squeeze(0).detach().cpu().numpy()
        if (not test) and (self.noise_process is not None):
            action = action + self.noise_process.step()
            action = np.clip(action, -1, 1)
        action = misc.tanh_expand(self.min_act, self.max_act, action)
        return action

    def step(self, state, action, reward, next_state, done):
        self.exp_index += 1
        action = misc.squish_tanh(self.min_act, self.max_act, action)
        self.replay_buffer.add(state, action, reward, next_state, done)

        if (len(self.replay_buffer) > self.batch_size and
            self.exp_index > self.warm_up and 
            self.exp_index % self.learn_every == 0):
            
            experience_batch = self.replay_buffer.sample()
            self.learn(experience_batch)
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            next_actions = self.policy_net_target(next_states)
            V_next_state = self.value_net_target(next_states, next_actions)
            V_target = rewards + (self.gamma * V_next_state * (1-dones))

        # update q network
        V_expected = self.value_net(states, actions)
        TD_error = (V_target - V_expected)
        value_loss = (TD_error).pow(2).mean()
        
        self.val_optimizer.zero_grad()
        (value_loss).backward()
        self.val_optimizer.step()

        # update policy
        policy_loss = -self.value_net(states, self.policy_net(states)).mean()
        self.pol_optimizer.zero_grad()
        policy_loss.backward()
        self.pol_optimizer.step()
        model_utils.soft_update_model(self.policy_net, 
            self.policy_net_target, self.tau)
        model_utils.soft_update_model(self.value_net, 
            self.value_net_target, self.tau)
        
    def save(self, fname):
        torch.save({"network": self.policy_net.state_dict()}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.policy_net.load_state_dict(dat["network"])
