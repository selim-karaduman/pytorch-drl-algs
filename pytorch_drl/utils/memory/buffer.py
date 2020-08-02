import math
import torch
import numpy as np
import random
import time
from collections import deque, namedtuple
from pytorch_drl.utils.memory.segment_tree import SumTree, MinTree

class UniformBuffer:
    def __init__(self, size, batch_size, seed, device, action_type=torch.long):
        self.size = size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
        self.Experience =  namedtuple('Experience', 
                                        ['state',
                                        'action',
                                        'reward',
                                        'next_state',
                                        'done'])
        self.buffer = deque(maxlen=size)
        self.action_type = action_type
        
    def add(self, state, action, reward, next_state, done):
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        device = self.device
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in idx:
            exp = self.buffer[i]
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).type(self.action_type).to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)
        
#-----------------------------------------------------------------------------

class PriorityBuffer(UniformBuffer):

    def __init__(self, size, batch_size, seed, device, 
                    action_type=torch.long, alpha=0.6, eps=1e-6):
        super().__init__(size, batch_size, seed, device)
        self.alpha = alpha
        self.eps = eps
        segment_tree_size = int(np.power(2, np.ceil(np.log2(size))))
        self.sum_tree = SumTree(segment_tree_size)
        self.min_tree = MinTree(segment_tree_size)
        self.max_priority = 1
        self.tree_index = 0

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.sum_tree[self.tree_index] = priority
        self.min_tree[self.tree_index] = priority
        self.tree_index = (self.tree_index + 1) % self.size
        
    def _update(self, ind, priority):
        priority_ =  priority ** self.alpha + self.eps
        assert(priority_ > 0)
        self.sum_tree[ind] = priority_
        self.min_tree[ind] = priority_
        self.max_priority = max(self.max_priority, priority)
        

    def update_indices(self, inds, priorities):
        for i in range(len(inds)):
            self._update(inds[i].item(), priorities[i].item())
        
    def sample(self, beta):
        device = self.device
        indices = self.sum_tree.sample_batch_idx(self.batch_size)
        weights = self.sum_tree[indices]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in indices:
            exp = self.buffer[i]
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)
        
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).type(self.action_type).to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        indices = torch.from_numpy(np.array(indices)).to(device)
        weights = torch.tensor(weights).float()
        
        p_total = self.sum_tree.get_sum()
        p_min = self.min_tree.get_min() / p_total
        max_weight = (p_min * len(self)) ** (-beta)
        
        weights = weights / p_total
        weights = (weights * len(self)) ** (-beta)
        weights = weights / max_weight
        weights = weights.unsqueeze(1).to(device)
        return (states, actions, rewards, next_states, dones, indices, weights)


#-----------------------------------------------------------------------------


class EpisodicBuffer:

    def __init__(self, size, seed, device, batch_size, action_type=torch.long):
        self.size = size
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.action_type = action_type
        self.buffer = deque(maxlen=size)
        

    def add(self, trajectory):
        """
        trajectory: [[s1,s2,..sn], [a..], [r..], [p..], [d...]]
        s: np.array
        a: int
        r: float
        p: np.array
        d: bool
        """
        self.buffer.append(trajectory)
        # Return this same trajectory in the same format as self.sample()
        return self.sample(batch=[trajectory])

    def sample(self, batch=None):
        device = self.device
        if batch is None:
            indices = np.random.choice(len(self.buffer), 
                                        self.batch_size, replace=False)
            batch = [self.buffer[ind] for ind in indices]
        
        batch = [map(np.vstack, tr) for tr in batch]
        states, actions, rewards, policies, dones = (
            map(lambda x: x.swapaxes(0,1), 
                map(np.stack, 
                    zip(*batch)))
            )
        """
        states: n_steps x batch_size x [state_size]
        """
        
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).type(self.action_type).to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        policies = torch.from_numpy(policies).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        return states, actions, rewards, policies, dones


    def __len__(self):
        return len(self.buffer)

