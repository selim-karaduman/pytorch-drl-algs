import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.algs.base import ActorCritic
from pytorch_drl.utils.parallel_env import ParallelEnv
from pytorch_drl.utils.schedule import LinearSchedule

class PPO(ActorCritic):

    def __init__(self, 
                 actor_critic, 
                 env_id,
                 epsilon_init=0.1, 
                 epsilon_final=0.1, 
                 epsilon_horizon=1,
                 gamma=0.99, 
                 epochs=4, 
                 lr=1e-3, 
                 tau=0.95,
                 n_env=8,
                 device="cpu",
                 normalize_rewards=False,
                 max_grad_norm=0.5,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 mini_batch_size=32,
                 gail=False,
                 use_gae=True,
                 ):
        super().__init__()

        self.gail = gail
        self.use_gae = use_gae
        self.on_policy_updates = False
        self.epsilon = LinearSchedule(epsilon_init, 
                                        epsilon_final, epsilon_horizon)
        self.gamma = gamma
        self.epochs = epochs
        self.actor_critic = actor_critic
        self.device = device
        self.tau = tau
        self.n_env = n_env
        self.normalize_rewards = normalize_rewards
        self.actor_critic.to(device)
        self.mini_batch_size = mini_batch_size
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.envs = ParallelEnv(env_id, n=n_env)
        self.action_space = self.envs.action_space
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        

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
     
    def _sample_action(self, state, grad):
            # Assumes only actor_critic combined models will be used
            with torch.set_grad_enabled(grad):
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
        state = self.cur_tr_step
        for i in range(tmax):
            state = torch.from_numpy(state).float().to(device)
            action, log_prob, critic_val, a_dist = \
                self._sample_action(state, grad=self.on_policy_updates)
            next_state, reward, done, _ = self.envs.step(
                                            action.cpu().numpy())
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device)) 
            dones.append(torch.FloatTensor(done).unsqueeze(1).to(device))
            values.append(critic_val)
            state = next_state
        
        self.cur_tr_step = state
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            actor_dist, final_v = self.actor_critic(state)
        values = values + [final_v]
        
        # For GAIL returns are not used
        if self.gail:
            return (torch.cat(log_probs), torch.cat(states), 
                    torch.cat(actions), torch.cat(values), torch.cat(dones))
        
        fut_ret = final_v.detach()
        advantage = 0
        v_targs = []
        advantages = []
        rets = []
        for t in reversed(range(len(rewards))):
            if self.use_gae:
                next_val = values[t + 1] * (1 - dones[t])
                delta = rewards[t] - values[t] + self.gamma * next_val
                advantage = (delta 
                                + advantage * self.gamma 
                                    * self.tau * (1 - dones[t]))
                v_targs.insert(0, advantage + values[t])
            else:
                fut_ret = rewards[t] + self.gamma * fut_ret * (1 - dones[t])
                v_targs.insert(0, fut_ret)
            advantages.insert(0, advantage)
        
        return (torch.cat(log_probs), torch.cat(states), torch.cat(actions), 
                    torch.cat(advantages), torch.cat(v_targs))


    def _clipped_surrogate_update(self, args):
        log_probs, states, actions, advantages, v_targs = args
        # Calculate actor_loss
        cur_dist, cur_val = self.actor_critic(states)
        cur_log_probs = cur_dist.log_prob(actions)
        ratio = (cur_log_probs - log_probs.detach()).exp()
        clip = torch.clamp(ratio, 1 - self.epsilon.value, 1 + self.epsilon.value)
        actor_loss = -torch.min(ratio * advantages, clip * advantages).mean()
        critic_loss = (cur_val - v_targs.detach()).pow(2).mean()
        entropy_loss = -cur_dist.entropy().mean()
        loss = (self.critic_coef * critic_loss 
                + actor_loss 
                + entropy_loss * self.entropy_coef)
        
        self.actor_critic.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                            self.max_grad_norm)
        self.optimizer.step()
        loss_val = loss.detach().mean().item()
        return loss_val

    def learn(self, args):
        # Sample
        log_probs, states, actions, advantages, v_targs = args
        batch_size = log_probs.shape[0]
        num_iters = batch_size // self.mini_batch_size
        loss = 0
        for e in range(self.epochs):
            indices = np.random.choice(np.arange(batch_size), 
                                        (num_iters, self.mini_batch_size), 
                                        replace=False)
            for i in range(num_iters):
                idx = indices[i]
                args_sampled = (log_probs[idx], states[idx], actions[idx], 
                    advantages[idx], v_targs[idx])
                loss += self._clipped_surrogate_update(args_sampled)
        return loss/(self.epochs*num_iters)

    def save(self, fname):
        torch.save({"actor_critic_sd": self.actor_critic.state_dict()}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.actor_critic.load_state_dict(dat["actor_critic_sd"])

