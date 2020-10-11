import torch
import numpy as np
from collections import deque
from pytorch_drl.algs.ppo import PPO
import torch.nn.functional as F

"""
PPO GAIL
"""
class GAIL:

    def __init__(self,
                 ppo_net_constr,
                 ppo_net_args,
                 discriminator_constr,
                 discriminator_args,
                 expert_trajectories,
                 env_id,
                 action_size,
                 gamma=0.99, 
                 gail_epochs=1,
                 ppo_epochs=4,
                 lr_ppo=None, 
                 lr_discriminator=None,
                 tau=0.95,
                 n_env=8,
                 device="cpu",
                 max_grad_norm=None,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 mini_batch_size=32,
                 normalize_rewards=True,
                 buf_len=10e4
                 ):
    
        self.ppo_net = ppo_net_constr(*ppo_net_args)
        self.ppo_net.to(device)
        self.discriminator = discriminator_constr(*discriminator_args)
        self.discriminator.to(device)
        self.device = device
        self.action_size = action_size
        self.env_id = env_id
        self.gamma = gamma
        self.tau = tau
        self.n_env = n_env
        self.expert_trajectories = expert_trajectories
        self.normalize_rewards = normalize_rewards
        self.gail_epochs = gail_epochs
        
        if lr_discriminator is None:
            self.optimizer = torch.optim.Adam(self.discriminator.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                 lr=lr_discriminator)

        self.ppo_agent = PPO(self.ppo_net, 
                             env_id,
                             gamma=gamma, 
                             epochs=ppo_epochs, 
                             lr=lr_ppo, 
                             tau=tau,
                             n_env=n_env,
                             device=device,
                             max_grad_norm=max_grad_norm,
                             critic_coef=critic_coef,
                             entropy_coef=entropy_coef,
                             mini_batch_size=mini_batch_size,
                             normalize_rewards=normalize_rewards)
        
    def act(self, state):
        return self.ppo_agent.act(state)

    def sample_real_pairs(self, size):
        indices = np.arange(self.expert_trajectories.shape[0])
        batch_indices = np.random.choice(indices, size=size, replace=False)
        batch = self.expert_trajectories[batch_indices]
        batch = torch.from_numpy(batch).float().to(self.device)
        return batch

    def train(self, tmax, n_traj, test_env, test_freq=1):
        batch_size = tmax * self.n_env
        d_loss = F.binary_cross_entropy
        losses = []
        scores = []
        last_scores = deque(maxlen=100)
        last_losses = deque(maxlen=100)
        for e in range(n_traj):
            log_probs, states, actions, values, dones = \
                self.ppo_agent.collect_trajectories(tmax, gail=True)

            real_pairs = self.sample_real_pairs(batch_size)
            actions_one_hot = torch.eye(self.action_size)[actions]\
                                .float().to(self.device)
            fake_pairs = torch.cat([states, actions_one_hot], -1)

            for i in range(self.gail_epochs):
                # Train discriminator
                self.discriminator.zero_grad()
                real_logits = self.discriminator(real_pairs)
                fake_logits = self.discriminator(fake_pairs)
                discriminator_loss = (
                    d_loss(real_logits, torch.ones_like(real_logits)) + 
                    d_loss(fake_logits, torch.zeros_like(fake_logits))
                    )
                discriminator_loss.backward()
                self.optimizer.step()

            # change returns with discriminator estimation
            fake_logits = self.discriminator(fake_pairs).detach()
            rewards = torch.log(fake_logits) - torch.log1p(-fake_logits)
            # GAE:
            fut_ret = values[-1]
            gae = 0
            advantages = []
            v_targs = []
            for t in reversed(range(rewards.shape[0])):
                fut_ret = rewards[t] + self.gamma * fut_ret * (1 - dones[t])
                next_val = values[t + 1] * (1 - dones[t])
                delta = rewards[t] - (values[t] - self.gamma * next_val)
                gae = delta + gae * self.gamma * self.tau * (1 - dones[t])
                advantages.insert(0, gae)
                v_targs.insert(0, gae + values[t])
            advantages = torch.cat(advantages)
            v_targs = torch.cat(v_targs)

            if self.normalize_rewards:
                mean_rew = advantages.mean()
                std_rew = advantages.std()
                advantages = (advantages - mean_rew)/(std_rew+1e-5)
            args = (log_probs, states, actions, advantages, rewards, v_targs)
            self.ppo_agent.learn(args)
            
            # ============================= TEST =============================
            if e % test_freq == 0:
                score = self.ppo_agent.test(test_env)
                scores.append(score)
                last_scores.append(score)
                avg_s = np.mean(last_scores)         
                print("\rAVG score is {}, i: {}".format(avg_s, e).ljust(48), 
                        end="")
                if avg_s >= 195.0:
                    print("Solved! Episode %d" %(i))
                    fname = "checkpoints/{}.pth".format("ppo_disc")
                    torch.save(self.ppo_net.state_dict(), fname)
                    break
        
        return scores, losses

    def test(self, env, render=False, n_times=1):
        self.ppo_agent.test(env, render=render, n_times=n_times)