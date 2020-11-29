import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.algs.base import ActorCritic
from pytorch_drl.utils.parallel_env import ParallelEnv

class A2C(ActorCritic):
    """
    tmax: Maximum number of steps in the 
            environment to avoid non-ending cases
    """
    def __init__(self, 
                 actor_critic, 
                 env_id,
                 gamma=0.99, 
                 lr=1e-3, 
                 gae_tau=0.95,
                 n_env=8,
                 device="cpu",
                 max_grad_norm=0.5,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 gail=False,
                 use_gae=False,
                 tmax=205,
                 ):

        super().__init__()
        self.gail = gail
        self.use_gae = use_gae
        self.gamma = gamma
        self.actor_critic = actor_critic
        self.device = device
        self.gae_tau = gae_tau
        self.n_env = n_env
        self.actor_critic.to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                             lr=lr)
        self.envs = ParallelEnv(env_id, n=n_env, seed=0, tmax=tmax)
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.action_space = self.envs.action_space

    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actor_dist, val = self.actor_critic(state)
        
        if deterministic:
            action = actor_dist.mode()
        else:
            action = actor_dist.sample()
        action = self.convert_to_numpy(action)
        return action
     
    def _sample_action(self, state):
            # Assumes only actor_critic combined models will be used
            actor_dist, critic_val = self.actor_critic(state)
            action = actor_dist.sample()
            log_prob = actor_dist.log_prob(action)
            # action: tensor of shape: (B, *action_space.shape)
            return action, log_prob, critic_val, actor_dist

    def collect_trajectories(self, tmax):
        device = self.device
        log_probs = []
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        entropies = []
        
        state = self.cur_tr_step
        for i in range(tmax):
            state = torch.from_numpy(state).float().to(device)
            action, log_prob, critic_val, a_dist = \
                self._sample_action(state)
            next_state, reward, done, _ = self.envs.step(
                                            action.cpu().numpy())
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device)) 
            dones.append(torch.FloatTensor(done).unsqueeze(1).to(device))
            values.append(critic_val)
            entropies.append(a_dist.entropy())
            state = next_state
        
        self.cur_tr_step = state
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            actor_dist, final_v = self.actor_critic(state)

        # For GAIL returns are not used
        if self.gail:
            return (torch.cat(log_probs), torch.cat(states), 
                    torch.cat(actions), torch.cat(values), torch.cat(dones))
        
        fut_ret = final_v.detach()
        advantage = 0
        v_targs = []
        rets = []
        for t in reversed(range(len(rewards))):
            if self.use_gae:
                if t == len(rewards) - 1:
                    next_val = final_v * (1 - dones[t])
                else:
                    next_val = values[t + 1] * (1 - dones[t])
                delta = rewards[t] - values[t] + self.gamma * next_val
                advantage = (delta 
                                + advantage * self.gamma 
                                    * self.gae_tau * (1 - dones[t]))
                v_targs.insert(0, advantage + values[t])
            else:
                fut_ret = rewards[t] + self.gamma * fut_ret * (1 - dones[t])
                v_targs.insert(0, fut_ret)
        
        return (torch.cat(log_probs), torch.cat(values),
                    torch.cat(v_targs), torch.cat(entropies))

    def learn(self, args):
        (log_probs, values, v_targs, entropies) = args
        advantages = v_targs.detach() - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = (advantages).pow(2).mean()
        entropy_loss = -entropies.mean()
        loss = (self.critic_coef * critic_loss 
                + actor_loss
                + self.entropy_coef * entropy_loss)
        self.actor_critic.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                            self.max_grad_norm)
        self.optimizer.step()
        return loss.detach().mean().item()

    def save(self, fname):
        torch.save({"actor_critic_sd": self.actor_critic.state_dict()}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.actor_critic.load_state_dict(dat["actor_critic_sd"])

