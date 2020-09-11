import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *
from pytorch_drl.utils.math import *
from pytorch_drl.utils.model_utils import *

class TRPO:

    def __init__(self, 
                 actor_constructor,
                 actor_args,
                 critic_constructor,
                 critic_args,
                 critic_use_bfgs=True,
                 critic_lr_sgd=1e-3,
                 critic_lr_bfgs=0.1,
                 critic_reg_bfgs=0,
                 bfgs_max_no_try=10,
                 max_kl=1e-2,
                 backtrack_alpha=0.8,
                 backtrack_steps=10,
                 damping_coeff=0.1,
                 env_name=None,
                 gamma=0.99, 
                 tau=0.95,
                 n_env=8,
                 device="cpu",
                 normalize_rewards=True,
                 max_grad_norm=0.5,
                 ):

        self.gamma = gamma
        self.damping_coeff = damping_coeff
        self.max_kl = max_kl
        self.backtrack_steps = backtrack_steps
        self.backtrack_alpha = backtrack_alpha
        self.bfgs_max_no_try = bfgs_max_no_try
        self.actor = actor_constructor(*actor_args) #policy returns: action distribution
        self.critic = critic_constructor(*critic_args)
        self.device = device
        self.tau = tau 
        self.n_env = n_env
        self.critic_lr_bfgs = critic_lr_bfgs
        self.normalize_rewards = normalize_rewards
        self.actor.to(device)
        self.critic.to(device)
        self.critic_use_bfgs = critic_use_bfgs
        self.critic_reg_bfgs = critic_reg_bfgs
        if critic_use_bfgs:
            self.critic_optimizer = torch.optim.LBFGS(self.critic.parameters(), 
                                                        lr=critic_lr_bfgs)
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                        lr=critic_lr_sgd)
        self.envs = ParallelEnv(env_name, n=n_env)
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        
    def act(self, state, deterministic=False):
        # state is a numpy array
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actor_dist = self.actor(state)
        if deterministic:
            action = actor_dist.probs.argmax(1).item()
        else:    
            action = actor_dist.sample().item()
        return action
    
    def _sample_action(self, state, no_grad=True):
        # state: batch_size x [state_size]
        with torch.no_grad():
            actor_dist = self.actor(state)
        action = actor_dist.sample()
        return action, actor_dist.log_prob(action)
    
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
            with torch.no_grad():
                critic_val = self.critic(state)
                action, log_prob = self._sample_action(state)
            next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
            
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            # rewards, dones shape: [n_games]
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device)) 
            dones.append(torch.FloatTensor(done).unsqueeze(1).to(device))
            values.append(critic_val)
            state = next_state
            
        # extend values for H+1
        self.cur_tr_step = state
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            final_v = self.critic(state)
        values = values + [final_v]
        
        # GAE:
        fut_ret = final_v.detach()
        gae = 0
        advantages = []
        v_targs = []
        # GAE for future rewards
        for t in reversed(range(len(rewards))):
            fut_ret = rewards[t] + self.gamma * fut_ret * (1 - dones[t])
            next_val = values[t + 1] * (1 - dones[t])
            
            delta = rewards[t] - (values[t] - self.gamma * next_val)
            gae = delta + gae * self.gamma * self.tau * (1 - dones[t])
            
            advantages.insert(0, gae)
            v_targs.insert(0, fut_ret)
            #: gae + value[t]
        
        advantages = torch.cat(advantages)
        if self.normalize_rewards:
            advantages = ((advantages - advantages.mean()) 
                            / (advantages.std() + 1e-8))
        
        return (torch.cat(log_probs), torch.cat(states), 
                torch.cat(actions), advantages, 
                torch.cat(rewards), torch.cat(v_targs))

    def learn(self, args):
        log_probs, states, actions, advantages, rewards, v_targs = args
        # Calculate actor_loss
        # ========================= LEARN FOR CRITIC =========================
        if self.critic_use_bfgs:
            def closure():
                self.critic.zero_grad()
                cur_val = self.critic(states)
                critic_loss = (cur_val - v_targs.detach()).pow(2).mean()
                # Regularizer 
                critic_loss += (self.critic_reg_bfgs 
                        * get_flat_parameters(self.critic, grad=True).pow(2).sum())
                critic_loss.backward()
                return critic_loss
            # TODO
        else:
            cur_val = self.critic(states)
            critic_loss = (cur_val - v_targs.detach()).pow(2).mean()
            self.critic.zero_grad()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                                self.max_grad_norm)
            critic_loss.backward()
            self.critic_optimizer.step()

        # ====================================================================

        # =============================== TRPO ===============================
        # Calculate the gradient of policy objective:
        cur_dist = self.actor(states)
        old_log_prob_sa = log_probs
        #print(old_log_prob_sa)
        #print(torch.exp(old_log_prob_sa))
        
        #asdf()
        log_prob_sa = cur_dist.log_prob(actions)
        objective = (torch.exp(log_prob_sa - old_log_prob_sa)
                        * advantages).mean()
        g = torch.autograd.grad(objective, self.actor.parameters())
        g = torch.cat([torch.flatten(p) for p in g]).detach()
        #print("graD:")
        #print(g)
        #print("--------------")
        #ffdd()
        def kl_divergence_local(network):
            dist = network(states)
            return (dist.probs.detach()
                    * (dist.logits.detach() - dist.logits)
                    ).sum(-1).mean()
            
        def kl_hvp(x):
            return hvp(kl_divergence_local, (self.actor,), self.actor, 
                        x, damping=self.damping_coeff)
        
        # solve for: Hessian(kl) x = g
        step_dir = conjugate_gradient(kl_hvp, g, max_steps=10)
        sHs = step_dir.T.dot(kl_hvp(step_dir)) * 0.5
        beta = torch.sqrt(sHs / self.max_kl)
        
        # Line search: shrink size until improvement 
        #   if none: no update
        step = step_dir / beta
        old_params = get_flat_parameters(self.actor)
        
        for i in range(self.backtrack_steps):
            set_parameters(self.actor, step + old_params)
            with torch.no_grad():
                new_dist = self.actor(states)
                new_log_prob_sa = new_dist.log_prob(actions)
                new_estimate = (torch.exp(new_log_prob_sa - old_log_prob_sa)
                                * advantages).mean()
                new_kl = (cur_dist.probs
                        * (cur_dist.logits 
                            - new_dist.logits)).sum(-1).mean()
                improvement = new_estimate - objective
            if (new_kl < self.max_kl * 1.5) and (improvement > 0):
                # print("Updated actor parameters")
                # Complete search; don't undo the step
                break
            step *= self.backtrack_alpha
        else:
            set_parameters(self.actor, old_params)

        # ====================================================================
        
        loss_ret = (critic_loss.data - objective.data)
        return loss_ret.cpu().detach().numpy()

    def train(self, tmax, n_traj, test_env, test_freq=4):
        losses = []
        scores = []
        last_scores = deque(maxlen=20)
        last_losses = deque(maxlen=20)
        for i in range(n_traj):
            args = self.collect_trajectories(tmax)
            avg_loss = self.learn(args)
            
            losses.append(avg_loss)
            last_losses.append(avg_loss)

            avg_loss_tot = np.mean(last_losses)
            
            if i % test_freq == 0:
                score = 0
                state = test_env.reset()
                while True:
                    action = self.act(state, deterministic=True)
                    state, reward, done, _ = test_env.step(action)
                    score += reward
                    if done:
                        break
                scores.append(score)
                last_scores.append(score)
                avg_score = np.mean(scores)
                print("\rEp: {}; Score: {}, Loss: {}".format(i, avg_score, avg_loss_tot))
                
                if avg_score >= 195.0:
                    print("Solved! Episode %d" %(i))
                    fname = "checkpoints/{}.pth".format("ppo_disc")
                    torch.save(self.ppo_net.state_dict(), fname)
                    break
        
        return scores, losses
