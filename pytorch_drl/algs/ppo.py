import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *

class PPO:

    def __init__(self, 
                 ppo_net, 
                 env_id,
                 gamma=0.99, 
                 epsilon_init=0.1, 
                 epsilon_final=0.1, 
                 epsilon_horizon=1, 
                 epochs=4, 
                 lr=None, 
                 tau=0.95,
                 n_env=8,
                 device="cpu",
                 normalize_rewards=False,
                 max_grad_norm=0.5,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 mini_batch_size=32
                 ):

        self.gamma = gamma
        self.epsilon = LinearSchedule(epsilon_init, epsilon_final, epsilon_horizon)
        self.epochs = epochs
        self.ppo_net = ppo_net
        self.device = device
        self.tau = tau
        self.n_env = n_env
        self.normalize_rewards = normalize_rewards
        self.ppo_net.to(device)
        self.mini_batch_size = mini_batch_size
        
        if lr is None:
            self.optimizer = torch.optim.Adam(self.ppo_net.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.ppo_net.parameters(), lr=lr)
        
        self.envs = ParallelEnv(env_id, n=n_env)
        self.cur_tr_step = self.envs.reset()
        self.max_grad_norm = max_grad_norm
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        
    def act(self, state):
        # state is a numpy array
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actor_dist, critic_val = self.ppo_net(state)
        action = actor_dist.sample().item()
        return action
    
    def _sample_action(self, state, no_grad=True):
        # state: batch_size x [state_size]
        with torch.no_grad():
            actor_dist, critic_val = self.ppo_net(state)
        action = actor_dist.sample()
        return action, actor_dist.log_prob(action), critic_val
    
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
            action, log_prob, critic_val = self._sample_action(state)
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
            actor_dist, final_v = self.ppo_net(state)
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
            mean_rew = advantages.mean()
            std_rew = advantages.std()
            advantages = (advantages - mean_rew)/(std_rew+1e-5)
        
        return (torch.cat(log_probs), torch.cat(states), 
                torch.cat(actions), advantages, 
                torch.cat(rewards), torch.cat(v_targs))

    def _clipped_surrogate_update(self, args):
        log_probs, states, actions, advantages, rewards, v_targs = args
        # Calculate actor_loss
        cur_dist, cur_val = self.ppo_net(states)
        cur_log_probs = cur_dist.log_prob(actions)
        ratio = (cur_log_probs - log_probs.detach()).exp()
        clip = torch.clamp(ratio, 1 - self.epsilon.value, 1 + self.epsilon.value)
        actor_loss = -torch.min(ratio * advantages, clip * advantages).mean()

        critic_loss = (cur_val - v_targs.detach()).pow(2).mean()
        entropy_loss = -cur_dist.entropy().mean()
        loss = (self.critic_coef * critic_loss 
                + actor_loss 
                + entropy_loss * self.entropy_coef)
        
        self.ppo_net.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.ppo_net.parameters(), 
                                            self.max_grad_norm)
        self.optimizer.step()
        loss_val = loss.detach().mean().item()
        return loss_val

    def learn(self, args):
        # Sample
        log_probs, states, actions, advantages, rewards, v_targs = args
        batch_size = log_probs.shape[0]
        num_iters = batch_size // self.mini_batch_size

        loss = 0
        for e in range(self.epochs):
            indices = np.random.choice(np.arange(batch_size), 
                                        (num_iters, self.mini_batch_size), 
                                        replace=True)
            for i in range(num_iters):
                idx = indices[i]
                args_sampled = (log_probs[idx], states[idx], actions[idx], 
                    advantages[idx], rewards[idx], v_targs[idx])
                loss += self._clipped_surrogate_update(args_sampled)
        return loss/(self.epochs*num_iters)

    def train(self, tmax, n_traj, test_env, test_freq=1):
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
            print("\rTrajectory %d, AVG. Loss %.2f" %(i, avg_loss_tot))
            
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
                    torch.save(self.ppo_net.state_dict(), fname)
                    break
        
        return scores, losses
