import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
import random
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *
from pytorch_drl.utils.exploration import *
from pytorch_drl.utils.memory.buffer import PriorityBuffer, UniformBuffer

class DDPG:

    def __init__(self, 
                policy_net=None,
                policy_net_target=None,
                value_net=None,
                value_net_target=None,
                gamma=0.99, 
                lr_val=None,
                lr_pol=None,
                prioritized_replay=False,
                buf_size=int(1e5),
                batch_size=64,
                is_beta=0.6,
                beta_horz=10e5,
                pr_alpha=0.2,
                tau=1e-3,
                device="cpu",
                normalize_rewards=False,
                max_grad_norm=0.5,
                n=1,
                nstep=False,
                min_act=float("-inf"),
                max_act=float("inf"),
                noise_process=None,
                warm_up=1e2,
                seed=0
                ):
        self.policy_net = policy_net
        self.policy_net_target = policy_net_target
        self.value_net = value_net
        self.value_net_target = value_net_target
        self.prioritized = prioritized_replay
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.tau = tau
        self.normalize_rewards = normalize_rewards
        self.max_grad_norm = max_grad_norm
        self.n = n
        self.nstep = nstep
        self.seed = random.seed(seed)
        self.nstep_buffer = deque(maxlen=n)
        self.is_beta = is_beta
        self.beta_horz = beta_horz
        self.pr_alpha = pr_alpha
        self.warm_up = warm_up
        action_type = torch.float

        self.min_act = min_act
        self.max_act = max_act
        self.noise_process = noise_process
        self.experience_index = 0
        self.discount = self.gamma ** n if nstep else self.gamma
        if (policy_net is None 
                or value_net is None
                or policy_net_target is None
                or value_net_target is None):
            raise ValueError

        value_net.to(device)        
        policy_net.to(device)
        value_net_target.to(device)        
        policy_net_target.to(device)

        if lr_val is None:
            self.val_optimizer = torch.optim.Adam(self.value_net.parameters())
        else:
            self.val_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                    lr=lr_val)

        if lr_pol is None:
            self.pol_optimizer = torch.optim.Adam(self.policy_net.parameters())
        else:
            self.pol_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                    lr=lr_pol)
    
        if self.prioritized:
            self.beta = LinearSchedule(is_beta, 1.0, beta_horz)
            self.replay_buffer = PriorityBuffer(buf_size,
                                                batch_size, seed, device,
                                                action_type=action_type,
                                                alpha=pr_alpha)
        else:
            self.replay_buffer = UniformBuffer(buf_size, batch_size,
                                                seed, device,
                                                action_type=action_type)

    def nn_to_action(self, action):
        # action: each element is between [-1, 1]
        return self.min_act + (self.max_act - self.min_act) * (action + 1) / 2

    def action_to_nn(self, action):
        return -1 + (action - self.min_act) / (self.max_act - self.min_act) * 2 

    def act(self, state, test=False):
        if len(self.replay_buffer) < self.warm_up:
            return np.random.uniform(self.min_act, self.max_act)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy_net(state)
        action = action.squeeze(0).detach().cpu().numpy()
        if (not test) and (self.noise_process is not None):
            action = action + self.noise_process.step()
        action = np.clip(action, -1, 1)
        return self.nn_to_action(action)

    def append_to_buffer(self, state, action, reward, next_state, done):
        if self.nstep:
            self.nstep_buffer.append((state, action, reward, next_state, done))

            if len(self.nstep_buffer) < self.n:
                return

            last_state = state
            first_state, first_action = self.nstep_buffer[0][:2]
            discounted_reward = 0
            for s, a, r, s_, d in reversed(self.nstep_buffer):
                discounted_reward = self.gamma * discounted_reward * (1-d) + r
                if d: 
                    last_state = s_
                    done = True

            self.replay_buffer.add(first_state, first_action, 
                                    discounted_reward, last_state, done)
        
        else:
            self.replay_buffer.add(state, action, reward, next_state, done)
        
    def step(self, state, action, reward, next_state, done):
        self.experience_index += 1
        action = self.action_to_nn(action)
        self.append_to_buffer(state, action, reward, next_state, done)
        if (len(self.replay_buffer) > self.batch_size and
            self.experience_index > self.warm_up):
            
            if self.prioritized:
                experience_batch = self.replay_buffer.sample(self.beta.value)
            else:
                experience_batch = self.replay_buffer.sample()
            self.learn(experience_batch)
        
        if self.prioritized:
            self.beta.step()

    def learn(self, experiences):
        if self.prioritized:
            states, actions, rewards,\
                next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
            next_actions = self.policy_net_target(next_states)
            V_next_state = self.value_net_target(next_states, next_actions)
            V_target = rewards + (self.discount * V_next_state * (1-dones))

        # update q network
        V_expected = self.value_net(states, actions)
        TD_error = (V_target - V_expected)
        
        if self.prioritized:
            value_loss = torch.abs(TD_error)
            new_priorities = value_loss.squeeze().detach().cpu().numpy()
            self.replay_buffer.update_indices(indices, new_priorities)
            value_loss = (weights * value_loss * 0.5).mean()
        else:
            value_loss = (TD_error).pow(2).mean()
        
        self.val_optimizer.zero_grad()
        (value_loss).backward()
        self.val_optimizer.step()

        # update policy
        policy_loss = -(self.value_net(states, self.policy_net(states)).mean())
        self.pol_optimizer.zero_grad()
        policy_loss.backward()
        self.pol_optimizer.step()
        self.soft_update_target(self.policy_net, self.policy_net_target)
        self.soft_update_target(self.value_net, self.value_net_target)
        
    def soft_update_target(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(),
                                  target_net.parameters()):
            target_param.data.copy_((1-self.tau)*target_param 
                                    + self.tau*source_param)