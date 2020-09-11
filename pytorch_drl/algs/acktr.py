import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.kfac import KFACOptimizer
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *

class ACKTR:

    def __init__(self, 
                 actor_critic_constr, 
                 actor_critic_args,
                 env_id,
                 gamma=0.99, 
                 lr=None, 
                 tau=0.95,
                 n_env=8,
                 device="cpu",
                 max_grad_norm=0.5,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 vf_fisher_coef=1.0
                 ):

        self.gamma = gamma
        self.network = actor_critic_constr(*actor_critic_args)
        self.device = device
        self.tau = tau
        self.n_env = n_env
        self.vf_fisher_coef = vf_fisher_coef
        self.network.to(device)
        
        if lr is None:
            self.optimizer = KFACOptimizer(self.network)
        else:
            self.optimizer = KFACOptimizer(self.network, lr=lr)
        
        self.envs = ParallelEnv(env_id, n=n_env)
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        
    def act(self, state, test=True):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actor_dist, critic_val = self.network(state)
        action = actor_dist.sample().item()
        return action
    
    def _sample_action(self, state):
        actor_dist, critic_val = self.network(state)
        action = actor_dist.sample()
        return action, actor_dist.log_prob(action), critic_val, actor_dist
    
    def learn(self, tmax):
        device = self.device
        log_probs = []
        states = []
        rewards = []
        dones = []
        actions = []
        values = []
        entropies = []
        
        state = self.cur_tr_step
        for i in range(tmax):
            state = torch.from_numpy(state).float().to(device)
            action, log_prob, critic_val, dist = self._sample_action(state)
            next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
            
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device)) 
            dones.append(torch.FloatTensor(done).unsqueeze(1).to(device))
            values.append(critic_val)
            entropies.append(dist.entropy())
            state = next_state
            
        self.cur_tr_step = state
        state = torch.from_numpy(state).float().to(device)
        actor_dist, final_v = self.network(state)
        
        fut_return = final_v.detach()
        returns = []
        for t in reversed(range(len(rewards))):
            fut_return = rewards[t] + self.gamma * fut_return * (1 - dones[t])
            returns.insert(0, fut_return)
        
        returns = torch.cat(returns)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        states = torch.cat(states)
        actions = torch.cat(actions)
        entropies = torch.cat(entropies)

        advantage = returns.detach() - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = (advantage).pow(2).mean()
        entropy_loss = -entropies.mean()

        if self.optimizer.steps % self.optimizer.Ts == 0:
            self.network.zero_grad()
            pg_fisher_loss = -(log_probs).mean()
            sample_net = (values + torch.randn_like(values)).detach()
            vf_fisher_loss = -(sample_net - values).pow(2).mean()
            joint_loss = (pg_fisher_loss 
                            + self.vf_fisher_coef * vf_fisher_loss)

            self.optimizer.acc_stats = True
            joint_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False


        loss = (self.critic_coef * critic_loss 
                + actor_loss
                + self.entropy_coef * entropy_loss)
        
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
  
        return loss.detach().mean().item()

    def train(self, tmax, n_traj, test_env, test_freq=1):
        losses = []
        scores = []
        last_scores = deque(maxlen=100)
        last_losses = deque(maxlen=100)
        next_state = None
        for i in range(n_traj):
            loss = self.learn(tmax)
            losses.append(loss)
            last_losses.append(loss)
            avg_loss_tot = np.mean(last_losses)
            # TEST:
            if i % test_freq == 0:
                score = 0
                state = test_env.reset()
                while True:
                    action = self.act(state)
                    state, reward, done, _ = test_env.step(action)
                    score += reward
                    if done:
                        break
                scores.append(score)
                last_scores.append(score)
                avg_s = np.mean(last_scores)         
                print("\rAVG score is {}, i: {}".format(avg_s, i).ljust(48), 
                        end="")
                if avg_s >= 195.0:
                    print("Solved! Episode %d" %(i))
                    fname = "checkpoints/{}.pth".format("acktr_disc")
                    torch.save(self.network.state_dict(), fname)
                    break
        return scores, losses
