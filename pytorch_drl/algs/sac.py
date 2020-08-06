import time
from torch.distributions import Normal
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

class SAC:

    def __init__(self, 
                policy_net_constructor=None,
                policy_net_args=None,
                value_net_constructor=None,
                value_net_args=None,
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
                min_act=float("-inf"),
                max_act=float("inf"),
                noise_process=None,
                warm_up=1e2,
                learn_interval=2,
                seed=0
                ):

        if (policy_net_constructor is None) or (value_net_constructor is None):
            print("Model constructors are required")
            raise ValueError

        self.policy_net = policy_net_constructor(*policy_net_args)
        
        self.value_net1 = value_net_constructor(*value_net_args)
        self.value_net_target1 = value_net_constructor(*value_net_args)
        self.value_net_target1.load_state_dict(self.value_net1.state_dict())

        self.value_net2 = value_net_constructor(*value_net_args)
        self.value_net_target2 = value_net_constructor(*value_net_args)
        self.value_net_target2.load_state_dict(self.value_net2.state_dict())

        self.buf_size = buf_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.discount = gamma
        self.device = device
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.seed = random.seed(seed)
        self.warm_up = warm_up
        self.learn_interval = learn_interval
        self.entropy_alpha = entropy_alpha
        action_type = torch.float
        self.experience_index = 0


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

    def nn_to_action(self, action):
        # action: each element is between [-1, 1]
        return self.min_act + (self.max_act - self.min_act) * (action + 1) / 2

    def action_to_nn(self, action):
        return -1 + (action - self.min_act) / (self.max_act - self.min_act) * 2 

    def act(self, state, test=False):
        if len(self.replay_buffer) < self.warm_up:
            return np.random.uniform(self.min_act, self.max_act)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        add_noise = (not test)
        with torch.no_grad():
            action, _ = self.policy_net(state, add_noise)
        action = action.squeeze(0).detach().cpu().numpy()
        return self.nn_to_action(action)

    def append_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def step(self, state, action, reward, next_state, done):
        self.experience_index += 1
        action = self.action_to_nn(action)
        self.append_to_buffer(state, action, reward, next_state, done)
        if (len(self.replay_buffer) > self.batch_size and
            self.experience_index > self.warm_up and
            self.experience_index % self.learn_interval == 0
            ):
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

        self.soft_update_target(self.value_net1, self.value_net_target1)
        self.soft_update_target(self.value_net2, self.value_net_target2)

    def soft_update_target(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(),
                                  target_net.parameters()):
            target_param.data.copy_((1-self.tau)*target_param 
                                    + self.tau*source_param)