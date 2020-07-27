import math
import random
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from pytorch_drl.models.rainbow_models import *
from pytorch_drl.utils.memory.buffer import PriorityBuffer, UniformBuffer
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *

class Rainbow:
   
    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 vmin=0,
                 vmax=0,
                 atoms=0,
                 distributional=False,
                 n_quants=0,
                 quantile_regression=False,
                 prioritized_replay=False,
                 ddqn=False,
                 n=1, #n-step
                 nstep=False,
                 noisy=False,
                 model1=None,
                 model2=None,
                 gamma = 0.99,
                 lr = None,
                 n_replay = 4,
                 buf_size = int(1e5),
                 batch_size = 64,
                 is_beta = 0.6, #importance sampling (per)
                 beta_horz = 10e5,
                 pr_alpha = 0.2,
                 tau = 1e-3,
                 device = "cpu",
                 eps_schedule=LinearSchedule(1.0, 0.01, 1e4)
                 ):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.prioritized = prioritized_replay
        self.ddqn = ddqn
        self.distributional = distributional
        self.n_quants = n_quants
        self.quantile_regression = quantile_regression
        self.n = n
        self.nstep = nstep
        self.noisy = noisy
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = atoms
        self.nstep_buffer = deque(maxlen=n)
        self.online_net = model1
        self.target_net = model2
        self.device = device
        self.n_replay = n_replay
        self.experience_index = 0
        self.eps_schedule = eps_schedule

        if quantile_regression:
            self.tau_hat = (torch.arange(n_quants, dtype=torch.float)/n_quants 
                            + 1/(n_quants*2))
        
        if distributional:
            self.support = torch.linspace(vmin, vmax, atoms).to(device)
            self.delta_z = (vmax - vmin) / (atoms - 1) 

        if model1 == None or model2 == None:
            raise ValueError
        
        if lr is None:
            self.optimizer = torch.optim.Adam(self.online_net.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.online_net.to(device)
        self.target_net.to(device)
        
        if self.nstep:
            self.discount = self.gamma ** n
        else:
            self.discount = self.gamma

        if self.prioritized:
            self.beta = LinearSchedule(is_beta, 1.0, beta_horz)
            self.replay_buffer = PriorityBuffer(buf_size,
                                                batch_size, seed, device,
                                                alpha=pr_alpha)
        else:
            self.replay_buffer = UniformBuffer(buf_size, batch_size,
                                               seed, device)
        
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
        self.append_to_buffer(state, action, reward, next_state, done)
        self.experience_index = (self.experience_index + 1) % self.n_replay
        if self.experience_index == 0\
            and len(self.replay_buffer) > self.batch_size:

            if self.prioritized:
                experience_batch = self.replay_buffer.sample(self.beta.value)
            else:
                experience_batch = self.replay_buffer.sample()

            self.learn(experience_batch)
            self.soft_update_target()
        
        if self.prioritized:
            self.beta.step()
                
    def get_best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if self.distributional:
            expected_vals = (self.online_net(state) * self.support).sum(2)
            action = expected_vals.argmax(1).item()
        elif self.quantile_regression:
            action = ((self.online_net(state).detach().mean(2))
                            .argmax(1).item())
        else:
            action = self.online_net(state).argmax(1).item()
        return action

    def act(self, state, test=False):
        eps = self.eps_schedule.value
        if (not self.noisy) and (not test) and (random.random() < eps):
            action = random.choice(np.arange(self.action_size))
        else:
            if test: # test
                self.online_net.eval()
            with torch.no_grad():
                action = self.get_best_action(state)
            if test:
                self.online_net.train()

        self.eps_schedule.step()
        return action

    def learn_expected(self, experiences):
        if self.prioritized:
            states, actions, rewards,\
                next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            if self.ddqn:
                next_state_actions = self.online_net(next_states)\
                                         .detach().argmax(1).unsqueeze(1)
                Q_next_state = self.target_net(next_states)\
                                   .detach().gather(1, next_state_actions)
            else:
                Q_next_state = self.target_net(next_states)\
                                   .detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.discount * Q_next_state * (1-dones))
        Q_expected = self.online_net(states).gather(1, actions)
        
        if self.prioritized:
            loss = torch.abs(Q_expected - Q_targets)
            new_priorities = loss.squeeze().detach().cpu().numpy()
            self.replay_buffer.update_indices(indices, new_priorities)
            loss = (weights * loss).mean()
        else:
            loss = (Q_expected - Q_targets).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_distributional(self, experiences):
        if self.prioritized:
            states, actions, rewards,\
                next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        
        log_z = self.online_net(states, log=True)
        log_z = log_z[range(self.batch_size), actions.squeeze()]
        with torch.no_grad():
            if self.ddqn:
                next_state_actions = (
                    (self.online_net(next_states).detach() * self.support)
                    .sum(2)
                    .argmax(1)
                )
                z_next = self.target_net(next_states)
                z_next = z_next[range(self.batch_size), 
                                                next_state_actions]
            else:
                net_output = self.target_net(next_states).detach()
                best_actions = (net_output * self.support).sum(2).argmax(1)
                z_next = net_output[range(self.batch_size), best_actions]
        Tz = rewards + (self.discount * self.support * (1-dones))
        Tz = Tz.clamp(min=self.vmin, max=self.vmax)
        b = (Tz - self.vmin) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        l[(l > 0) * (l == u)] -= 1
        u[(u < self.atoms - 1) * (l == u)] += 1
        m = states.new_zeros(self.batch_size, self.atoms)
        
        offsets = torch.linspace(
                0, (self.batch_size - 1) * self.atoms, self.batch_size
            ).long()\
            .unsqueeze(1)\
            .expand(self.batch_size, self.atoms)\
            .to(actions)
        
        m.view(-1).index_add_(
            0, (l + offsets).view(-1), (z_next * (u.float() - b)).view(-1)
        )
        m.view(-1).index_add_(
            0, (u + offsets).view(-1), (z_next * (b - l.float())).view(-1)
        )
        
        if self.prioritized:
            loss = -torch.sum((m * log_z), 1)
            new_priorities = loss.squeeze().detach().cpu().numpy()
            self.replay_buffer.update_indices(indices, new_priorities)
            loss = (weights * loss).mean()
        else:
            loss = -torch.sum((m * log_z), 1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_qr(self, experiences):
        if self.prioritized:
            states, actions, rewards,\
                next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        qnt_sa = self.online_net(states)
        qnt_sa = qnt_sa[range(self.batch_size), actions.squeeze()]

        with torch.no_grad():
            if self.ddqn:
                next_state_actions = (
                    (self.online_net(next_states).detach().mean(2))
                    .argmax(1)
                )
                qnt_next = self.target_net(next_states)
                qnt_next = qnt_next[range(self.batch_size), 
                                                next_state_actions]
            else:
                net_output = self.target_net(next_states).detach()
                best_actions = (net_output.mean(2)).argmax(1)
                qnt_next = net_output[range(self.batch_size), best_actions]
        qnt_tg = rewards + (self.discount * qnt_next * (1-dones))

        diff = qnt_tg.unsqueeze(1) - qnt_sa.unsqueeze(2) #Get for all points
        loss = quantile_huber_loss(diff, self.tau_hat.view(1, -1, 1), k=1)
        
        loss = loss.mean(1).sum(-1)
        if self.prioritized:
            new_priorities = loss.squeeze().detach().cpu().numpy()
            self.replay_buffer.update_indices(indices, new_priorities)
            loss = (weights * loss).mean()
        else:
            loss =  loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, experiences): 
        if self.quantile_regression:
            self.learn_qr(experiences)
        elif self.distributional:
            self.learn_distributional(experiences)
        else:
            self.learn_expected(experiences)

        if self.noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

    def soft_update_target(self,):
        for param1, param2 in zip(self.online_net.parameters(),
                                  self.target_net.parameters()):
            param2.data.copy_((1-self.tau)*param2 + self.tau*param1)

