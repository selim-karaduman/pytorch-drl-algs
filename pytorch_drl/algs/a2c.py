import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *

class A2C:

    def __init__(self, 
                 a2c_net, 
                 env_id,
                 gamma=0.99, 
                 lr=None, 
                 tau=0.95,
                 n_env=8,
                 device="cpu",
                 max_grad_norm=0.5,
                 critic_coef=0.5,
                 entropy_coef=0.001,
                 ):

        self.gamma = gamma
        self.a2c_net = a2c_net
        self.device = device
        self.tau = tau
        self.n_env = n_env
        self.a2c_net.to(device)
        
        if lr is None:
            self.optimizer = torch.optim.Adam(self.a2c_net.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=lr)
        
        self.envs = ParallelEnv(env_id, n=n_env)
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        
    def act(self, state):
        # state is a numpy array
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actor_dist, critic_val = self.a2c_net(state)
        action = actor_dist.sample().item()
        return action
    
    def _sample_action(self, state, no_grad=True):
        # state: batch_size x [state_size]
        actor_dist, critic_val = self.a2c_net(state)
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
        entropy = 0
        
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
            entropy += (dist.entropy().mean())
            state = next_state
            
        self.cur_tr_step = state
        state = torch.from_numpy(state).float().to(device)
        actor_dist, final_v = self.a2c_net(state)
        
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

        
        
        cur_dist, cur_val = self.a2c_net(states)
        cur_log_prob = cur_dist.log_prob(actions)
        advantage = returns.detach() - cur_val
        actor_loss = -(cur_log_prob * advantage.detach()).mean()
        critic_loss = (advantage).pow(2).mean()
        entropy_loss = -cur_dist.entropy().mean()

        """"
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = (values - v_targs.detach()).pow(2).mean()
        entropy_loss = -entropy
        """
        loss = (self.critic_coef * critic_loss 
                + actor_loss
                + 0.01 * entropy_loss )
        
        self.a2c_net.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.a2c_net.parameters(), 
                                            self.max_grad_norm)
        self.optimizer.step()
        loss_val = loss.detach().mean().item()
        return loss_val

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
            print("\rTrajectory %d, AVG. Loss %.2f" %(i, avg_loss_tot))
            
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
                avg_score = np.mean(last_scores)         
                print("\rTEST at {}; score is {}".format(i, score))
                print("\rAVG score is {}".format(avg_score))

                if avg_score >= 195.0:
                    print("Solved! Episode %d" %(i))
                    fname = "checkpoints/{}.pth".format("ppo_disc")
                    torch.save(self.a2c_net.state_dict(), fname)
                    break
        
        return scores, losses
