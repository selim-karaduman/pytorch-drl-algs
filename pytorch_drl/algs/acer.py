import time
from torch.distributions import Categorical
import torch
from torch import nn
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import *
from pytorch_drl.utils.loss import *
from pytorch_drl.utils.parallel_env import *
from pytorch_drl.utils.model_utils import *
from pytorch_drl.utils.memory.buffer import *
from pytorch_drl.utils.shared_optim import AdamShared
import torch.multiprocessing as mp

class ACER_Agent(mp.Process):

    def __init__(self, 
                 shared_model=None,
                 average_model=None,
                 queue=None,
                 net_constr=None,
                 net_args=None,
                 env_name=None,
                 env_constr=None,
                 env_args=None,
                 gamma=0.99, 
                 replay_n=4,
                 lr=None, 
                 n_env=8,
                 normalize_rewards=False,
                 polyak_alpha=0.99,
                 trpo_theta=1,
                 use_trpo=True,
                 entropy_coefficient=1e-4,
                 memory_size_steps=100_000,
                 max_episodes=1_000,
                 max_episode_length=200,
                 max_traj_length=100,
                 start_off_policy=2000,
                 clip=10,
                 batch_size=16,
                 max_grad_norm=None,
                 seed=0,
                 mp_id=-1,
                 optimizer=None
                 ):

        super().__init__()
        self.max_episodes = max_episodes
        self.net_constr = net_constr
        self.trpo_theta = trpo_theta
        self.clip = clip
        self.net_args = net_args
        self.env_name = env_name
        self.max_traj_length = max_traj_length
        self.gamma = gamma
        self.polyak_alpha = polyak_alpha
        self.replay_n = replay_n
        self.lr = lr
        self.entropy_coefficient = entropy_coefficient
        self.device = "cpu"
        self.normalize_rewards = normalize_rewards
        self.use_trpo = use_trpo
        self.max_grad_norm = max_grad_norm
        self.queue = queue
        self.n_env = n_env
        self.mp_id = mp_id
        self.shared_model = shared_model
        self.average_model = average_model
        self.max_episode_length = max_episode_length
        self.start_off_policy = start_off_policy

        self.n_eps = 1e-10 # epsilon for log, div
        self.model = net_constr(*net_args)
        self.model.load_state_dict(self.shared_model.state_dict())
        self.optimizer = optimizer

        memory_size = memory_size_steps // max_traj_length // n_env
        self.buffer = EpisodicBuffer(memory_size, seed, self.device, batch_size)
        
        self.env = env_constr(*env_args)
        self.state = self.env.reset()
        self.episode_t = 0
        self.episode_score = 0

    
    # Collects an episode and returns
    #    the data in the same format as buffer.sample
    def collect_episode(self):
        states, actions, rewards, policies, dones = [], [], [], [], []
        for i in range(self.max_traj_length-1):
            self.episode_t += 1
            state_th = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
            policy, q_value = self.model(state_th)
            action = Categorical(policy).sample().item()
            next_state, reward, done, _ = self.env.step(action)
            done = done or (self.episode_t > self.max_episode_length)
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            policies.append(policy.squeeze(0).detach().cpu().numpy())
            self.state = next_state
            self.episode_score += reward

            if done or self.episode_t > self.max_episode_length:
                self.episode_t = 0
                self.state = self.env.reset()
                self.queue.put(self.episode_score)
                self.episode_score = 0

        states.append(self.state)
        episode = [states, actions, rewards, policies, dones]
        return self.buffer.add(episode)

    def learn(self, offline=True):
        self.model.load_state_dict(self.shared_model.state_dict())
        eps = self.n_eps

        if offline:
            batch = self.buffer.sample()
        else:
            batch = self.collect_episode()
            
        states, actions, rewards, b_policies, dones = batch
        policies, q_values, avg_policies = [], [], []

        # ====================== COLLECT DATA ================================
        k = states.shape[0]
        for i in range(k):
            state = states[i]
            policy, q_value = self.model(state)
            avg_policy, _ = self.average_model(state)
            avg_policy.detach_()
            policies.append(policy)
            q_values.append(q_value)
            avg_policies.append(avg_policy)

        policies = torch.stack(policies)
        q_values = torch.stack(q_values)
        avg_policies = torch.stack(avg_policies)

        # ========================= LEARN ====================================
        
        value = (policies[k-1] * q_values[k-1]).sum(-1, keepdims=True).detach()
        q_ret = value
        loss = 0
        for t in reversed(range(k-1)):
            value = (policies[t] * q_values[t]).sum(-1, keepdims=True).detach()
            q_ret = rewards[t] + self.gamma * q_ret * (1-dones[t])
            ratio = (policies[t] / (b_policies[t]+eps)).detach()
            correction_constant = ((1 - self.clip/(ratio+eps)).clamp(min=0) 
                                    * policies[t]).detach()
            policy_log = (policies[t]+eps).log()
            delta = q_ret - q_values[t].gather(1, actions[t])

            policy_loss = -(ratio.gather(1, actions[t]).clamp(max=self.clip) 
                            * policy_log.gather(1, actions[t])
                            * (q_ret - value)
                            ).mean()
            policy_loss += -(correction_constant 
                            * policy_log 
                            * (q_values[t].detach() - value)).sum(-1).mean()

            critic_loss = ((delta).pow(2) / 2).mean()

            # =========================== TRPO ===============================
            if self.use_trpo:
                k_grad = (avg_policies[t] / (policies[t]+eps)).detach()
                gradient = -torch.autograd.grad(inputs=policies, 
                                                outputs=policy_loss,
                                                retain_graph=True)[0][t]
                k_dot_g = (k_grad * gradient).sum(-1).mean()
                k_dot_k = (k_grad * k_grad).sum(-1).mean()
                # the vector that will be subtracted from the gradient
                grad_offset_norm = ((k_dot_g - self.trpo_theta) 
                                    / k_dot_k).clamp(min=0)
                # kl divergence
                kl_div = (avg_policies[t] 
                            * ((avg_policies[t] + eps).log() 
                                - (policies[t] + eps).log())).sum(-1).mean()
                policy_trpo_loss = (grad_offset_norm * kl_div)
                policy_loss += policy_trpo_loss

            # ================================================================
            entropy_loss = -(Categorical(policies[t]).entropy() 
                                * self.entropy_coefficient).mean()

            loss += (critic_loss + policy_loss + entropy_loss)
            q_ret = (ratio.gather(1, actions[t]).clamp(max=1) 
                    * (delta).detach()
                    + value
                    )

        # update average model and shared model, 
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                            self.max_grad_norm)
        
        transfer_gradients(self.model, self.shared_model)
        self.optimizer.step()
        soft_update_model(self.shared_model, 
                self.average_model, 1 - self.polyak_alpha)
        

    # runs in parallel
    def run(self):
        for i in range(self.max_episodes):
            self.learn(offline=False)
            if i < self.start_off_policy:
                continue

            n = int(np.random.exponential(self.replay_n))
            for e in range(n):
                self.learn()
        
        self.queue.put("done")
        self.env.close()


class ACER:

    def __init__(self, *args, **kwargs):
        
        if "net_constr" not in kwargs:
            print("A model constructor that suits ACER implementation is needed")
            raise ValueError

        if "env_constr" not in kwargs or "env_name" not in kwargs:
            print("Environment is required")
            raise ValueError
        
        net_constr = kwargs["net_constr"]
        net_args = kwargs["net_args"]
        self.shared_model = net_constr(*net_args)
        self.shared_model.share_memory()
        
        self.average_model = net_constr(*net_args)
        self.average_model.share_memory()
        self.average_model.load_state_dict(self.shared_model.state_dict())
        self.queue = mp.Queue()
        lr = kwargs["lr"] if "lr" in kwargs else 1e-3
        self.optimizer = AdamShared(self.shared_model.parameters(), lr=lr)

        self.n_env = kwargs["n_env"]
        self.args = args
        self.kwargs = kwargs
        self.kwargs["shared_model"] = self.shared_model
        self.kwargs["average_model"] = self.average_model
        self.kwargs["queue"] = self.queue
        self.kwargs["optimizer"] = self.optimizer

        self.env_constr = kwargs["env_constr"]
        self.env_args = kwargs["env_args"]

    def train(self, max_score, alg_name="acer", tmax=200):
        agents = [ACER_Agent(*self.args, **self.kwargs, mp_id=i)\
                     for i in range(self.n_env)]
        [agent.start() for agent in agents]
        scores = []
        last_scores = deque(maxlen=max(100, self.n_env*25))
        done = 0
        i = 0
        while True:    
            msg = self.queue.get()
            if msg != "done": # msg is score
                i += 1
                scores.append(msg)
                last_scores.append(msg)
                avg_score = np.mean(last_scores)
                print("\rScore: ", msg, end="")
                if avg_score >= max_score:
                    print("\nSolved!")
                    [agent.terminate() for agent in agents]
                    break
            else:
                done += 1
                if done == self.n_env:
                    break
        [agent.join() for agent in agents]

        """
        Save the model
        """
        fname = "checkpoints/{}.pth".format(alg_name)
        torch.save(self.shared_model.state_dict(), fname)
        return scores


    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        policy, q_value = self.shared_model(state)
        if deterministic:
            action = policy.argmax(-1).item()
        else:
            action = Categorical(policy).sample().item()
        return action

    def test(self, n_episodes, max_t, render=True, deterministic=False):
        # test
        env = self.env_constr(*self.env_args)
        for i in range(n_episodes):
            score = 0
            state = env.reset()
            for i in range(max_t):
                action = self.act(state, deterministic)
                state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                score += reward
                if done:
                    break
            print(score)
        env.close()