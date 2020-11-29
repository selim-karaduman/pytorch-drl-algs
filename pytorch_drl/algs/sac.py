import time
from torch.distributions import Normal
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

class SAC(ValueBased):

    def __init__(self, 
                 policy_net=None,
                 value_net1=None,
                 value_net2=None,
                 policy_noise=0.2,
                 entropy_alpha=0.2,
                 gamma=0.99, 
                 lr_val=1e-3,
                 lr_pol=1e-3,
                 buf_size=int(1e5),
                 batch_size=64,
                 tau=1e-3,
                 device="cpu",
                 max_grad_norm=0.5,
                 min_act=None,
                 max_act=None,
                 warm_up=1e2,
                 learn_every=2,
                 seed=0):
        super().__init__()
        self.policy_net = policy_net
        self.value_net1 = value_net1
        self.value_net_target1 = copy.deepcopy(value_net1)
        self.value_net2 = value_net2
        self.value_net_target2 = copy.deepcopy(value_net2)
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.discount = gamma
        self.device = device
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.seed = random.seed(seed)
        self.warm_up = warm_up
        self.learn_every = learn_every
        self.entropy_alpha = entropy_alpha
        action_type = torch.float
        self.min_act = min_act
        self.max_act = max_act
        self.value_net1.to(device) 
        self.value_net_target1.to(device)
        self.value_net2.to(device) 
        self.value_net_target2.to(device)
        self.policy_net.to(device)
        self.val_optimizer1 = torch.optim.Adam(self.value_net1.parameters(),
                                                    lr=lr_val)
        self.val_optimizer2 = torch.optim.Adam(self.value_net2.parameters(),
                                                    lr=lr_val)
        self.pol_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                    lr=lr_pol)
        
        self.replay_buffer = UniformBuffer(buf_size, batch_size,
                                            seed, device,
                                            action_type=action_type)

    def act(self, state, test=False):
        if len(self.replay_buffer) < self.warm_up:
            return np.random.uniform(self.min_act, self.max_act)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        add_noise = (not test)
        with torch.no_grad():
            self.policy_net.eval()
            action, _ = self.policy_net(state, add_noise)
            self.policy_net.train()
        action = action.squeeze(0).detach().cpu().numpy()
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
            next_actions, next_logp = self.policy_net(next_states)
            V_next_state = torch.min(
                            self.value_net_target1(next_states, next_actions),
                            self.value_net_target2(next_states, next_actions),
                            )
            V_next_state -= self.entropy_alpha * next_logp
            V_target = rewards + (self.discount * V_next_state * (1-dones))

        # update q network
        V_expected1 = self.value_net1(states, actions)
        V_expected2 = self.value_net2(states, actions)
        TD_error1 = (V_target - V_expected1)
        TD_error2 = (V_target - V_expected2)
        
        value_loss1 = (TD_error1).pow(2).mean()
        value_loss2 = (TD_error2).pow(2).mean()
    
        self.val_optimizer1.zero_grad()
        (value_loss1).backward()
        self.val_optimizer1.step()
        
        self.val_optimizer2.zero_grad()
        (value_loss2).backward()
        self.val_optimizer2.step()

        # update policy
        sample_actions, logp = self.policy_net(states)
        policy_loss = -(
                self.value_net1(states, sample_actions)
                - self.entropy_alpha * logp
                ).mean()
        self.pol_optimizer.zero_grad()
        policy_loss.backward()
        self.pol_optimizer.step()

        model_utils.soft_update_model(self.value_net1, 
            self.value_net_target1, self.tau)
        model_utils.soft_update_model(self.value_net2, 
            self.value_net_target2, self.tau)

    def save(self, fname):
        torch.save({"network": self.policy_net.state_dict()}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.policy_net.load_state_dict(dat["network"])
