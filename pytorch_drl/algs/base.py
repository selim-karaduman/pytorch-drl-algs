import gym
from collections import deque
import torch
from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):

    def __init__(self):
        self.exp_index = 0

    @abstractmethod
    def save(self, fname):
        pass

    @abstractmethod
    def load(self, fname):
        pass

# ======================================================================

class ActorCritic(Agent):

    @abstractmethod
    def act(self, state, deterministic=False):
        pass

    @abstractmethod
    def _sample_action(self, state, grad):
        """
        Sample action together with other tensors in  a tuple. 
            - Action is always the first element
        :rtype: tuple
        """
        pass

    @abstractmethod
    def collect_trajectories(self, tmax):
        """
        :rtype: tuple
        """
        pass

    @abstractmethod
    def learn(self, args):
        """
        :rtype: tuple
        """
        pass

    def convert_to_numpy(self, action):
        # return appropiate formats
        if isinstance(self.action_space, gym.spaces.Box):
            action = action.squeeze(0).detach().cpu().numpy()
        elif isinstance(self.action_space, gym.spaces.Discrete):
            action = action.item()
        elif isinstance(self.action_space, gym.spaces.MultiBinary): 
            action = action.squeeze(0).detach().cpu().numpy()
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete): 
            action = action.squeeze(0).detach().cpu().numpy()
        return action
    

    def train(self, tmax, n_episodes, test_env, max_score, 
                model_name, test_freq=1, det_test=False):
        losses = []
        scores = []
        last_scores = deque(maxlen=100)
        last_losses = deque(maxlen=100)
        for i in range(n_episodes):
            args = self.collect_trajectories(tmax)
            avg_loss = self.learn(args) 
            losses.append(avg_loss)
            last_losses.append(avg_loss)
            avg_loss_tot = np.mean(last_losses)
            if i % test_freq == 0:
                score = self.test(test_env)
                scores.append(score)
                last_scores.append(score)
                avg_s = np.mean(last_scores)        
                print("\rAvg score: {:.2f} i: {}".format(avg_s, i).ljust(48),
                         end="")
                if avg_s >= max_score:
                    print("Solved! Episode %d" %(i))
                    fname = "checkpoints/{}.dat".format(model_name)
                    #self.save(fname)
                    break
        return scores, losses

    def test(self, env, render=False, n_episodes=1):
        score_avg = 0
        for i in range(n_episodes):
            state = env.reset()
            score = 0
            while True:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                score += reward
                if done:
                    break
            score_avg += score
            if render:
                print(score)
        env.close()
        return score_avg/n_episodes

    # For GAIL
    def save_trajectories(self, n_steps, fname, action_size):
        states = []
        actions = []
        state = self.envs.reset()
        for i in range(n_steps):
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().to(self.device)
                action = self._sample_action(state_t, grad=False)[0]
            action_np = action.detach().cpu().numpy()
            next_state, reward, done, _ = self.envs.step(action_np)
            states.append(state)
            if isinstance(self.action_space, gym.spaces.Discrete):
                # onehot encoding
                actions.append(np.eye(action_size)[action_np])
            else:    
                actions.append(action_np)
            state = next_state
            if i % 50 == 0:
                print("\r{} / {}".format(i, n_steps).ljust(48), end="")

        states = np.concatenate(states)
        actions = np.concatenate(actions)
        trajectory = np.concatenate([states, actions], axis=-1)
        with open(fname, 'wb') as f:
            np.save(f, trajectory)
        return trajectory

# ======================================================================

class ValueBased(Agent):
    
    @abstractmethod
    def act(self, state, test=False):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def learn(self, experiences):
        pass

    def train_episode(self, env, max_t, render=False):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = self.act(state)
            if render:
                env.render()
            next_state, reward, done, _ = env.step(action)
            self.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                return score
        return score

    def train(self, env, tmax, n_episodes, alg_name, max_score, 
                render_freq=None, test_freq=None, save_models=True):
        scores = []  
        scores_window = deque(maxlen=100)
        test_scores = []
        test_scores_window = deque(maxlen=100)
        for i_episode in range(1, n_episodes+1):
            render = (render_freq and (i_episode % render_freq == 0))
            score = self.train_episode(env, tmax, render)
            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)

            if (test_freq is not None) and (i_episode % test_freq == 0):
                t_score = self.test(env)
                test_scores.append(t_score)
                test_scores_window.append(t_score)
                print("Avg Test score: ", np.mean(test_scores_window))

            print("\rAvg score: {:.2f} i: {}".format(avg_s, i).ljust(48),
                         end="")
            if avg_score >= max_score:
                print("Solved! Episode {}".format(i_episode))
                if save_models:
                    fname = "checkpoints/{}.pth".format(alg_name)
                    torch.save(agent.online_net.state_dict(), fname)
                break
        env.close()
        return scores

    def test(self, env, render=False, n_episodes=1):
        score_avg = 0
        for i in range(n_episodes):
            state = env.reset()
            score = 0
            while True:
                action = self.act(state, test=True)
                state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                score += reward
                if done:
                    break
            score_avg += score
            if render:
                print(score)
        env.close()
        return score_avg/n_episodes
    

