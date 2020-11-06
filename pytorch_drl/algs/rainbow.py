import math
import random
import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from pytorch_drl.utils.memory.buffer import PriorityBuffer, UniformBuffer
from pytorch_drl.utils.schedule import ExpSchedule, LinearSchedule
import pytorch_drl.utils.loss as loss_functions
import pytorch_drl.utils.model_utils as model_utils
from pytorch_drl.algs.base import ValueBased

class Rainbow(ValueBased):
   
    def __init__(self,
                 action_size,
                 model=None,
                 gamma=0.99,
                 lr=1e-3,
                 learn_every=4,
                 buf_size=int(1e5),
                 batch_size=64,
                 tau=1e-3,
                 device="cpu",
                 seed=0,
                 eps_start=1, 
                 eps_final=0.01, 
                 eps_n=1e5, #number of steps till for epsilon decay
                 ddqn=False, #double q learning

                 categorical_dqn=False, #use c51 algortihm
                 vmin=0, #categorical_dqn: vmin
                 vmax=0, #categorical_dqn: vmax
                 atoms=51,#categorical_dqn: atoms
                 
                 quantile_regression=False, # use quantile regression
                 n_quants=51, #quantile_regression: number of quants
                 
                 prioritized_replay=False, # use per
                 is_beta=0.6, # per: importance sampling
                 beta_horz=1e5, #per: beta
                 pr_alpha=0.2, # per: alpha
                 
                 nstep=False, #use nstep returns
                 n=1, #n-step: n
                 
                 noisy=False, #use nosiy linear layers
                 ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.prioritized = prioritized_replay
        self.ddqn = ddqn
        self.categorical_dqn = categorical_dqn
        self.n_quants = n_quants
        self.quantile_regression = quantile_regression
        self.n = n
        self.nstep = nstep
        self.noisy = noisy
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = atoms
        self.nstep_buffer = deque(maxlen=n)
        self.online_net = model
        self.target_net = copy.deepcopy(model)
        self.device = device
        self.learn_every = learn_every
        self.online_net.to(device)
        self.target_net.to(device)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.eps_schedule = ExpSchedule(eps_start, eps_final, eps_n)

        if quantile_regression:
            self.tau_hat = (torch.arange(n_quants, dtype=torch.float)/n_quants 
                            + 1 / (n_quants * 2)).to(device)
        
        if categorical_dqn:
            self.support = torch.linspace(vmin, vmax, atoms).to(device)
            self.delta_z = (vmax - vmin) / (atoms - 1) 
        
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
            self.nstep_buffer.append((state, action, reward,
                                        next_state, done))
            if len(self.nstep_buffer) < self.n:
                return
            # calculate n-step return:
            last_state = next_state
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
        self.exp_index = (self.exp_index + 1) % self.learn_every
        if self.exp_index == 0\
            and len(self.replay_buffer) > self.batch_size:

            if self.prioritized:
                experience_batch = self.replay_buffer.sample(self.beta.value)
            else:
                experience_batch = self.replay_buffer.sample()

            self.learn(experience_batch)
        
        if self.prioritized:
            self.beta.step()
                
    def get_best_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if self.categorical_dqn:
            expected_vals = (self.online_net(state) * self.support).sum(2)
            action = expected_vals.argmax(1).item()
        elif self.quantile_regression:
            expected_vals = (self.online_net(state).detach().mean(2))
            action = expected_vals.argmax(1).item()
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

    def learn(self, experiences): 
        if self.quantile_regression:
            self.learn_qr(experiences)
        elif self.categorical_dqn:
            self.learn_categorical_dqn(experiences)
        else:
            self.learn_expected(experiences)

        if self.noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

        model_utils.soft_update_model(self.online_net, 
                                        self.target_net, self.tau)

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

    def learn_categorical_dqn(self, experiences):
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
        loss = loss_functions.quantile_huber_loss(diff, 
                    self.tau_hat.view(1, -1, 1), k=1)
        
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

    def save(self, fname):
        torch.save({"network": self.online_net.state_dict()}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.online_net.load_state_dict(dat["network"])