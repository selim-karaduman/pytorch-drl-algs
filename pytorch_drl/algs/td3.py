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

class TD3:

    def __init__(self, 
                policy_net_constructor=None,
                policy_net_args=None,
                value_net_constructor=None,
                value_net_args=None,
                policy_noise=0.2,
                noise_clip=0.5,
                gamma=0.99, 
                lr_val=1e-3,
                lr_pol=1e-3,
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
                update_interval=2,
                seed=0
                ):

        if (policy_net_constructor is None) or (value_net_constructor is None):
            print("Model constructors are required")
            raise ValueError

        self.policy_net = policy_net_constructor(*policy_net_args)
        self.policy_net_target = policy_net_constructor(*policy_net_args)
        self.policy_net_target.load_state_dict(self.policy_net.state_dict())

        self.value_net1 = value_net_constructor(*value_net_args)
        self.value_net_target1 = value_net_constructor(*value_net_args)
        self.value_net_target1.load_state_dict(self.value_net1.state_dict())

        self.value_net2 = value_net_constructor(*value_net_args)
        self.value_net_target2 = value_net_constructor(*value_net_args)
        self.value_net_target2.load_state_dict(self.value_net2.state_dict())

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
        self.update_interval = update_interval
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        action_type = torch.float
        self.experience_index = 0

        self.min_act = min_act
        self.max_act = max_act
        self.noise_process = noise_process
        self.discount = self.gamma ** n if nstep else self.gamma
        
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
        action = np.clip(action, -1, 1) # assumption: -1, 1 output from nn
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
        
        if self.prioritized:
            value_loss1 = torch.abs(TD_error1)
            value_loss2 = torch.abs(TD_error2)
            new_priorities = ((value_loss1 + value_loss2)/2)\
                                .squeeze().detach().cpu().numpy()
            self.replay_buffer.update_indices(indices, new_priorities)
            value_loss1 = (weights * value_loss1 * 0.5).mean()
            value_loss2 = (weights * value_loss2 * 0.5).mean()
        else:
            value_loss1 = (TD_error1).pow(2).mean()
            value_loss2 = (TD_error2).pow(2).mean()
        
        self.val_optimizer1.zero_grad()
        (value_loss1).backward()
        self.val_optimizer1.step()
        
        self.val_optimizer2.zero_grad()
        (value_loss2).backward()
        self.val_optimizer2.step()

        if self.experience_index % self.update_interval == 0:
            # update policy
            policy_loss = -(self.value_net1(states, self.policy_net(states)).mean())
            self.pol_optimizer.zero_grad()
            policy_loss.backward()
            self.pol_optimizer.step()

            self.soft_update_target(self.policy_net, self.policy_net_target)
            self.soft_update_target(self.value_net1, self.value_net_target1)
            self.soft_update_target(self.value_net2, self.value_net_target2)

    def soft_update_target(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(),
                                  target_net.parameters()):
            target_param.data.copy_((1-self.tau)*target_param 
                                    + self.tau*source_param)