import time
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
import pytorch_drl.utils.model_utils as model_utils
from pytorch_drl.utils.memory.buffer import MABuffer

"""
MADDPG:
- Only local
- No ensemble/inferred_policies
"""
#Discrete

class _DDPG:

    def __init__(self,
                 id=None,
                 policy_net=None,
                 policy_net_target=None,
                 value_net=None,
                 value_net_target=None,
                 gamma=0.99, 
                 lr_val=1e-3,
                 lr_pol=1e-3,
                 batch_size=1024,
                 device="cpu",
                 max_grad_norm=0.5
                 ):

        self.id = id
        self.policy_net = policy_net
        self.policy_net_target = policy_net_target
        self.value_net = value_net
        self.value_net_target = value_net_target 
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.policy_net.to(device)
        self.policy_net_target.to(device)
        self.value_net.to(device)
        self.value_net_target.to(device)
        self.val_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                lr=lr_val)
        self.pol_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                lr=lr_pol)
        
    def act(self, state, test=False, batched=False, 
            grad=False, use_th=False, target=False):
        if not batched:
            self.policy_net.eval()
            self.policy_net_target.eval()
            state = torch.from_numpy(state)\
                        .float().unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(grad):
            if target:
                x = self.policy_net_target(state)
            else:
                x = self.policy_net(state)
        
        if test:
            a = torch.argmax(x, dim=-1, keepdims=True)
            onehot = torch.zeros_like(x)
            onehot.scatter_(1, a, 1)    
        else:
            onehot = F.gumbel_softmax(x, hard=True)
        
        if not batched:
            self.policy_net.train()
            self.policy_net_target.train()
            onehot = onehot.squeeze(0)
        
        if not use_th:
            return onehot.cpu().numpy()
        
        return onehot
        
    def learn(self, experiences, agents):
        """
        experiences: tuple of list of tensors:
        experiences: (states, actions, rewards, next_states, dones)
            states: list of tensors
                [torch.tensor, torch.tensor,...] of size n_agents
                each tensor is of shape (B, state_size)

        """
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            next_actions = [agent.act(next_states[i], batched=True, 
                                        use_th=True, test=True, target=True)
                             for i, agent in enumerate(agents)]
            xa_n = torch.cat(next_states+next_actions, dim=1)
            V_next_state = self.value_net_target(xa_n)
            V_target = rewards[self.id] \
                        + (self.gamma * V_next_state * (1-dones[self.id]))
        
        xa = torch.cat(states+actions, dim=1)
        V_expected = self.value_net(xa)
        TD_error = (V_target - V_expected)
        value_loss = (TD_error).pow(2).mean()
        self.val_optimizer.zero_grad()
        (value_loss).backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 
                                    self.max_grad_norm)
        self.val_optimizer.step()

        cur_action = self.act(states[self.id], grad=True, batched=True, 
                                use_th=True)
        # update policy
        p_actions = [actions[i] if i!=self.id else cur_action 
                        for i, agent in enumerate(agents)]
        xa = torch.cat(states+p_actions, dim=1)
        policy_loss = -(self.value_net(xa).mean())
        self.pol_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                    self.max_grad_norm)
        self.pol_optimizer.step()

    
class MADDPG:

    def __init__(self, 
                policy_nets=None,
                policy_net_targets=None,
                value_nets=None,
                value_net_targets=None,
                gamma=0.99, 
                lr_val=1e-3,
                lr_pol=1e-3,
                buf_size=int(1e6),
                batch_size=1024,
                tau=1e-3,
                device="cpu",
                max_grad_norm=0.5,
                warm_up=1e2,
                env=None,
                seed=0,
                learn_every=100,
                ):

        self.policy_nets = policy_nets
        self.policy_net_targets = policy_net_targets
        self.value_nets = value_nets
        self.value_net_targets = value_net_targets
        self.warm_up = warm_up
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.tau = tau
        self.env = env
        self.learn_every = learn_every
        self.n_agents = len(policy_nets)
        self.step_id = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.buffer = MABuffer(self.buf_size, self.batch_size,
                                self.seed, self.device,
                                action_type=torch.float) #disc
        self.agents = [_DDPG(id=i,
                             policy_net=self.policy_nets[i],
                             policy_net_target=self.policy_net_targets[i],
                             value_net=self.value_nets[i],
                             value_net_target=self.value_net_targets[i],
                             gamma=gamma, 
                             lr_val=lr_val,
                             lr_pol=lr_pol,
                             batch_size=batch_size,
                             device=device,
                             max_grad_norm=max_grad_norm) 
                        for i in range(self.n_agents)]
        
    def train(self, n_episodes, max_t, test_every=1, max_score=0):
        env = self.env
        scores = []
        test_scores_window = deque(maxlen=100)
        for e in range(n_episodes):
            states = env.reset()
            score = np.zeros(self.n_agents)
            for t in range(max_t):
                actions = [agent.act(states[i]) 
                            for i, agent in enumerate(self.agents)]
                next_states, rewards, dones, _ = self.env.step(actions)
                self.buffer.add(states, actions, rewards, next_states, dones)
                states = next_states
                score = np.array(rewards)
                scores.append(score)
                self.step_id += 1
                if ((self.step_id % self.learn_every == 0) 
                        and (len(self.buffer) > self.batch_size)
                        and (self.step_id >= self.warm_up)):
                    for agent in self.agents:
                        batch = self.buffer.sample()
                        agent.learn(batch, self.agents)
                    for agent in self.agents:
                        model_utils.soft_update_model(agent.value_net, 
                            agent.value_net_target, self.tau)
                        model_utils.soft_update_model(agent.policy_net, 
                            agent.policy_net_target, self.tau)
                if dones[0]:
                    break

            if e % test_every == 0:
                test_score = self.test(max_t, 1, render=False)
                test_scores_window.append(test_score)
                avg_test_score = np.mean(test_scores_window, axis=0)
                print("\rAverage test score: {}, e: {}".ljust(48)\
                            .format(avg_test_score, e), end="")
                if np.mean(avg_test_score) >= max_score:
                    print("Solved! Episode {}".format(e))
                    # TODO save the models 
                    break
        return np.stack(scores)

    def test(self, max_t, n_episodes, render=True):
        env = self.env
        scores = []
        for e in range(n_episodes):
            states = env.reset()
            score = np.zeros(env.n)
            for t in range(max_t):
                actions = [agent.act(states[i], test=True) 
                            for i, agent in enumerate(self.agents)]
                next_states, rewards, dones, _ = self.env.step(actions)
                states = next_states
                score += rewards
                scores.append(score)
                if render:
                    env.render()
                if dones[0]:
                    break
        env.close()
        if render:
            print("Scores avg: ", np.mean(scores, axis=0))
        return np.mean(scores, axis=0)
