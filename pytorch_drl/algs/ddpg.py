import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
import random
import copy
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *
from pytorch_drl.utils.exploration import *
from pytorch_drl.utils.memory.buffer import PriorityBuffer, UniformBuffer
import pytorch_drl.utils.misc
import pytorch_drl.utils.model_utils

class DDPG:

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
                normalize_rewards=False,
                max_grad_norm=0.5,
                noise_process=None,
                warm_up=1e2,
                seed=0,
                env_id=None
                ):

        self.policy_net = policy_net
        self.policy_net_target = copy.deepcopy(policy_net)
        self.value_net = value_net
        self.value_net_target = copy.deepcopy(value_net)
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.tau = tau
        self.normalize_rewards = normalize_rewards
        self.max_grad_norm = max_grad_norm
        self.seed = random.seed(seed)
        self.warm_up = warm_up
        self.env = env
        self.action_space = gym.make(env_id).action_space
        self.noise_process = noise_process
        if isinstance(self.action_space, gym.spaces.Box):
            self.min_act = self.action_space.low
            self.max_act = self.action_space.high
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.action_size = self.action_space.n
        else:
            print("action space should be Discrete or Box")
        
        self.experience_index = 0
        value_net.to(device)        
        policy_net.to(device)
        value_net_target.to(device)        
        policy_net_target.to(device)
        self.val_optimizer = torch.optim.Adam(value_net.parameters(), 
                                                lr=lr_val)
        self.pol_optimizer = torch.optim.Adam(policy_net.parameters(),
                                                lr=lr_pol)
        self.replay_buffer = UniformBuffer(buf_size, batch_size, seed, 
                                            device, action_type=torch.float)

    def act(self, state, test=False):
        if len(self.replay_buffer) < self.warm_up: 
            return self.action_space.sample()

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy_net(state)
        if ((not test) and isinstance(self.action_space, gym.spaces.Box)):
            action = action + self.noise_process.step()
            action = np.clip(action, -1, 1)
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = misc.onehot_to_index(action, self.action_size)
        elif isinstance(self.action_space, gym.spaces.Box):
            action = misc.tanh_expand(self.min_act, self.max_act, action)
        action = action.squeeze(0).detach().cpu().numpy()
        return action

    def step(self, state, action, reward, next_state, done):
        self.experience_index += 1
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = misc.index_to_onehot(action, )
        elif isinstance(self.action_space, gym.spaces.Box):
            action = misc.squish_tanh(self.min_act, self.max_act, action)

        self.replay_buffer.add(state, action, reward, next_state, done)
        if (len(self.replay_buffer) > self.batch_size and
                self.experience_index > self.warm_up):
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
        
    def train():
        asdf
    def test():
        adf