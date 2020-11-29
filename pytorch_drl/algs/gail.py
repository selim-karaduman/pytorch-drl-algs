import torch
import numpy as np
from collections import deque
from pytorch_drl.algs.ppo import PPO
import torch.nn.functional as F
import pytorch_drl.utils.misc as misc

"""
PPO GAIL
This is experimental: ppo.py is used, 
    which is implemented with the assumption that 
    the environment will signal the reward.
"""
class GAIL:

    def __init__(self,
                 actor_critic,
                 discriminator,
                 expert_trajectories,
                 env_id,
                 action_size,
                 gamma=0.99, 
                 gail_epochs=1,
                 ppo_epochs=4,
                 lr_ppo=None, 
                 lr_discriminator=1e-3,
                 gae_tau=0.95,
                 n_env=8,
                 device="cpu",
                 max_grad_norm=None,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 mini_batch_size=32,
                 normalize_rewards=True,
                 buf_len=10e4,
                 tmax=200,
                 ):
    
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.discriminator = discriminator
        self.discriminator.to(device)
        self.device = device
        self.action_size = action_size
        self.env_id = env_id
        self.gamma = gamma
        self.gae_tau = gae_tau
        self.n_env = n_env
        self.expert_trajectories = expert_trajectories
        self.normalize_rewards = normalize_rewards
        self.gail_epochs = gail_epochs
        
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=lr_discriminator)

        self.ppo_agent = PPO(self.actor_critic, 
                             env_id,
                             gamma=gamma, 
                             epochs=ppo_epochs, 
                             lr=lr_ppo, 
                             gae_tau=gae_tau,
                             n_env=n_env,
                             device=device,
                             max_grad_norm=max_grad_norm,
                             critic_coef=critic_coef,
                             entropy_coef=entropy_coef,
                             mini_batch_size=mini_batch_size,
                             normalize_rewards=normalize_rewards,
                             gail=True,
                             tmax=tmax)
        
    def act(self, state):
        return self.ppo_agent.act(state)

    def sample_real_pairs(self, size):
        indices = np.arange(self.expert_trajectories.shape[0])
        batch_indices = np.random.choice(indices, size=size, replace=False)
        batch = self.expert_trajectories[batch_indices]
        batch = torch.from_numpy(batch).float().to(self.device)
        return batch

    def train(self, tmax, n_traj, test_env, max_score, alg_name, test_freq=1):
        batch_size = tmax * self.n_env
        d_loss = F.binary_cross_entropy
        losses = []
        scores = []
        last_scores = deque(maxlen=100)
        last_losses = deque(maxlen=100)
        for e in range(n_traj):
            args = self.ppo_agent.collect_trajectories(tmax)
            # only ppo
            log_probs, states, actions, values, dones = args
            
            real_pairs = self.sample_real_pairs(batch_size)
            actions_one_hot = misc.index_to_onehot(actions, self.action_size)\
                                    .float().to(self.device)
            fake_pairs = torch.cat([states, actions_one_hot], -1)

            for i in range(self.gail_epochs):
                # Train discriminator
                self.discriminator.zero_grad()
                real_probs = self.discriminator(real_pairs)
                fake_probs = self.discriminator(fake_pairs)
                discriminator_loss = (
                    #d_loss(real_probs, torch.ones_like(real_probs)) + 
                    #d_loss(fake_probs, torch.zeros_like(fake_probs))
                    d_loss(real_probs, torch.zeros_like(real_probs)) + 
                    d_loss(fake_probs, torch.ones_like(fake_probs))
                    )
                discriminator_loss.backward()
                self.optimizer.step()

            # change returns with discriminator estimation
            fake_probs = self.discriminator(fake_pairs).detach()
            rewards = -torch.log(fake_probs)

            # GAE:
            gae = 0
            advantages = []
            v_targs = []
            for t in reversed(range(rewards.shape[0])):
                next_val = values[t + 1] * (1 - dones[t])
                delta = rewards[t] - (values[t] - self.gamma * next_val)
                gae = delta + gae * self.gamma * self.gae_tau * (1 - dones[t])
                advantages.insert(0, gae)
                v_targs.insert(0, gae + values[t])
            advantages = torch.cat(advantages)
            v_targs = torch.cat(v_targs)
            # log_probs, states, actions, advantages, v_targs = args
            args = (log_probs, states, actions, advantages, v_targs)
            self.ppo_agent.learn(args)
            
            # ============================= TEST =======================
            if e % test_freq == 0:
                score = self.ppo_agent.test(test_env)
                scores.append(score)
                last_scores.append(score)
                avg = np.mean(last_scores)         
                print("\rAvg score is {:.2f}, i: {}".format(avg, e).ljust(48), 
                        end="")
                if avg >= max_score:
                    print("Solved! Episode %d" %(i))
                    fname = "checkpoints/{}.pth".format(alg_name)
                    self.save(fname)
                    break
        
        return scores, losses

    def test(self, env, render=False, n_episodes=1):
        self.ppo_agent.test(env, render=render, n_episodes=n_episodes)

    def save(self, fname):
        self.ppo_agent.save(fname)

    def load(self, fname):
        self.ppo_agent.load(fname)
