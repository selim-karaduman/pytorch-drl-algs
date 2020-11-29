import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
import random
import copy
from collections import deque
import pytorch_drl.utils.model_utils as model_utils
import pytorch_drl.utils.misc as misc
from pytorch_drl.utils.memory.buffer import PriorityBuffer, UniformBuffer
from pytorch_drl.algs.base import ValueBased

class TD3(ValueBased):

    def __init__(self, 
                 policy_net=None,
                 value_net1=None,
                 value_net2=None,
                 policy_noise=0.2,
                 noise_clip=0.5,
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
                 noise_process=None,
                 warm_up=1e2,
                 policy_delay=2,
                 learn_every=1,
                 seed=0):
        super().__init__()
        self.policy_net = policy_net
        self.policy_net_target = copy.deepcopy(policy_net)
        self.value_net1 = value_net1
        self.value_net_target1 = copy.deepcopy(value_net1)
        self.value_net2 = value_net2
        self.value_net_target2 = copy.deepcopy(value_net2)
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.device = device
        self.tau = tau
        self.learn_every = learn_every
        self.max_grad_norm = max_grad_norm
        self.seed = random.seed(seed)
        self.warm_up = warm_up
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        action_type = torch.float
        self.min_act = min_act
        self.max_act = max_act
        self.noise_process = noise_process
        self.discount = self.gamma
        self.value_net1.to(device) 
        self.value_net_target1.to(device)
        self.value_net2.to(device) 
        self.value_net_target2.to(device)
        self.policy_net.to(device)
        self.policy_net_target.to(device)
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
            noise = (torch.randn(*actions.shape) * self.policy_noise)\
                    .clamp(-self.noise_clip, self.noise_clip)\
                    .float().to(self.device)
            next_actions = self.policy_net_target(next_states) + noise
            V_next_state = torch.min(
                            self.value_net_target1(next_states, next_actions),
                            self.value_net_target2(next_states, next_actions),
                            ).detach()
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

        if self.exp_index % self.policy_delay == 0:
            # update policy
            policy_loss = -self.value_net1(states, 
                                self.policy_net(states)).mean()
            self.pol_optimizer.zero_grad()
            policy_loss.backward()
            self.pol_optimizer.step()

            model_utils.soft_update_model(self.policy_net, 
                self.policy_net_target, self.tau)
            model_utils.soft_update_model(self.value_net1, 
                self.value_net_target1, self.tau)
            model_utils.soft_update_model(self.value_net2, 
                self.value_net_target2, self.tau)

    def save(self, fname):
        torch.save({"network": self.policy_net.state_dict()}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.policy_net.load_state_dict(dat["network"])
