import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.algs.base import ActorCritic
from pytorch_drl.utils.parallel_env import ParallelEnv
import pytorch_drl.utils.math as math_utils
import pytorch_drl.utils.model_utils as model_utils

class TRPO(ActorCritic):

    def __init__(self, 
                 actor,
                 critic,
                 critic_lr=1e-3,
                 max_kl=1e-2,
                 backtrack_alpha=0.5,
                 backtrack_steps=10,
                 damping_coeff=0.1,
                 env_id=None,
                 gamma=0.99, 
                 gae_tau=0.95,
                 n_env=8,
                 device="cpu",
                 max_grad_norm=0.5,
                 gail=False,
                 use_gae=True,
                 ):
        super().__init__()

        self.gail = gail
        self.use_gae = use_gae        
        self.gamma = gamma
        self.damping_coeff = damping_coeff
        self.max_kl = max_kl
        self.backtrack_steps = backtrack_steps
        self.backtrack_alpha = backtrack_alpha
        self.actor = actor
        self.critic = critic
        self.device = device
        self.gae_tau = gae_tau 
        self.n_env = n_env
        self.actor.to(device)
        self.critic.to(device)
        self.critic_optimizer = torch.optim.Adam(
                                    self.critic.parameters(), 
                                    lr=critic_lr
                                    )
        self.envs = ParallelEnv(env_id, n=n_env)
        self.action_space = self.envs.action_space
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        
    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actor_dist = self.actor(state)
        
        if deterministic:
            action = actor_dist.mode()
        else:
            action = actor_dist.sample()
        action = self.convert_to_numpy(action)
        return action
    
    def _sample_action(self, state, grad):
        # Assumes only actor_critic combined models will be used
        actor_dist = self.actor(state)
        action = actor_dist.sample()
        log_prob = actor_dist.log_prob(action)
        # action: tensor of shape: (B, *action_space.shape)
        return action, log_prob, actor_dist

    def collect_trajectories(self, tmax):
        device = self.device
        log_probs = []
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        distributions_cls = None
        
        state = self.cur_tr_step
        for i in range(tmax):
            state = torch.from_numpy(state).float().to(device)
            action, log_prob, a_dist = \
                self._sample_action(state)
            critic_val = self.critic(state)
            next_state, reward, done, _ = self.envs.step(
                                            action.cpu().numpy())
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device)) 
            dones.append(torch.FloatTensor(done).unsqueeze(1).to(device))
            values.append(critic_val)
            if not distributions_cls:
                distributions_cls = a_dist.__class__
            state = next_state
        
        self.cur_tr_step = state
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            final_v = self.critic(state)
        
        # For GAIL returns are not used
        if self.gail:
            return (torch.cat(log_probs), torch.cat(states), 
                    torch.cat(actions), torch.cat(values), torch.cat(dones))
        
        fut_ret = final_v.detach()
        advantage = 0
        v_targs = []
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
        log_probs = torch.cat(log_probs)
        distributions = distributions_cls(log_probs)
        return (log_probs, torch.cat(states), torch.cat(actions), 
                    torch.cat(values), torch.cat(v_targs), distributions)
    
    def learn(self, args):
        log_probs, states, actions, values, v_targs, distributions = args
        advantages = v_targs.detach() - values
        # ===================== LEARN FOR CRITIC =======================
        critic_loss = (advantages).pow(2).mean()
        self.critic.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                            self.max_grad_norm)
        self.critic_optimizer.step()
        # ==============================================================
        advantages = ((advantages - advantages.mean()) 
                        / (advantages.std() + 1e-8)).detach()
        # =========================== TRPO =============================
        # Calculate the gradient of policy objective:
        old_log_probs = log_probs.detach()
        objective = (torch.exp(log_probs - old_log_probs)
                        * advantages).mean()
        #objective: pi(a|s)/pi(a|s).stop_grad() * A
        g = torch.autograd.grad(objective, self.actor.parameters())
        g = torch.cat([torch.flatten(p) for p in g]).detach()
        
        def kl_divergence_local():
            dist = self.actor(states)
            return (dist.probs.detach()
                    * (dist.logits.detach() - dist.logits)
                    ).sum(-1, keepdims=True).mean()
            
        def kl_hvp(x):
            return math_utils.hvp(kl_divergence_local, self.actor, x, 
                                    damping=self.damping_coeff)
        
        # solve for: Hessian(kl) x = g
        step_dir = math_utils.conjugate_gradient(kl_hvp, g, 
                                                    max_steps=10)
        sHs = (step_dir.T.dot(kl_hvp(step_dir)) * 0.5)
        beta = torch.sqrt(sHs / self.max_kl)
        
        # Line search: shrink size until improvement 
        step = step_dir / beta
        old_params = model_utils.get_flat_parameters(self.actor)

        # refer to: https://github.com/ikostrikov/
        #               pytorch-trpo/blob/master/trpo.py for linesearch
        # d_objective/d_param * delta_param:
        expected_improvement = (g * step).sum() 
        for i in range(self.backtrack_steps):
            model_utils.set_parameters(self.actor, step + old_params)
            with torch.no_grad():
                new_dist = self.actor(states)
                new_log_probs = new_dist.log_prob(actions)
                new_estimate = (torch.exp(new_log_probs - old_log_probs)
                                * advantages).mean()
                improvement = new_estimate - objective
            if (improvement/expected_improvement > 0.1) and (improvement > 0):
                break
            step *= self.backtrack_alpha
            expected_improvement *= self.backtrack_alpha
        else:
            model_utils.set_parameters(self.actor, old_params)

        # ==============================================================
        
        loss_ret = (critic_loss.data - objective.data)
        return loss_ret.cpu().detach().numpy()

    def save(self, fname):
        torch.save({"critic_sd": self.critic.state_dict(),
                    "actor_sd": self.actor.state_dict(),}, fname)

    def load(self, fname):
        dat = torch.load(fname)
        self.actor.load_state_dict(dat["actor_sd"])
        self.critic.load_state_dict(dat["critic_sd"])

