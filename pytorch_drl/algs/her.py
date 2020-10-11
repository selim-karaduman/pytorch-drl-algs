import math
import random
import time
import torch
import gym
import numpy as np
from collections import deque


class HER:

    def __init__(self, env, alg_constr, alg_args, alg_kwargs):
        self.agent = alg_constr(*alg_args, **alg_kwargs)
        self.env = env
        self.concat = lambda s, g: np.concatenate([s, g], axis=-1)
        if not hasattr(self.agent, 'step'):
            print("""Only Off-policy algorithms can be used. 
                Algorithms to use: DQN(Rainbow file), DDPG, TD3, SAC.""")
            raise ValueError
        
    def train(self, n_traj, t_max, n_epochs, max_score, render_freq=None, 
                test_freq=10, save_models=False, checkpoint_name=None, 
                strategy=None, n_strategy=10): 
        
        if strategy == 'final':
            strategy = self.final_state_strategy
        elif strategy == 'random':
            strategy = self.random_strategy
        elif strategy == 'next':
            strategy = self.next_strategy
        else:
            strategy = self.random_bias_strategy
        
        scores = []  
        scores_window = deque(maxlen=100)
        test_scores = []
        test_scores_window = deque(maxlen=100)
        for i_episode in range(1, n_traj+1):
            render = (render_freq is not None) and (i_episode % render_freq == 0)
            score = self.train_episode(t_max, n_epochs, n_strategy, strategy, 
                                        render=render)
            
            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)
            if (test_freq is not None) and (i_episode % test_freq == 0):
                t_score = self.test(t_max, render=False)
                test_scores.append(t_score)
                test_scores_window.append(t_score)
                print("Avg Test score: ", np.mean(test_scores_window))

            print("\rEpisode %d, AVG. Score %.2f" %(i_episode, avg_score))
            if avg_score >= max_score:
                print("Solved! Episode %d" %(i_episode))
                if save_models:
                    fname = "checkpoints/{}.pth".format(checkpoint_name)
                    torch.save(agent.online_net.state_dict(), fname)
                break
        self.env.close()
        return scores

    def final_state_strategy(self, episode, state, n):
        # Take final state as the goal
        if not hasattr(self, 'cache'):
            self.cache = [list(zip(*episode))[0][-1]]
        return self.cache

    def random_bias_strategy(self, episode, state, n):
        return random.choices(list(zip(*episode))[0], k=n) + [state]

    def random_strategy(self, episode, state, n):
        return random.choices(list(zip(*episode))[0], k=n)

    def next_strategy(self, episode, state, n):
        return [state]        

    def train_episode(self, t_max, n_epochs, n_strategy, strategy, 
                        render=False):
        """
        strategy(episode, state): returns list of goals
        """
        concat = self.concat
        episode = []
        state, goal = self.env.reset()
        score = 0
        # ============================= HER =============================
        for t in range(t_max):
            action = self.agent.act(concat(state, goal))
            if render:
                self.env.render()
            (next_state, next_goal), reward, done, info = self.env.step(action)
            episode.append((state, goal, action, 
                            reward, next_state, done, info))
            state = next_state
            goal = next_goal
            score += reward
            if done:
                break
        
        for state, goal, action, reward, next_state, done, info in episode:
            self.agent.append_to_buffer(concat(state, goal), action, reward, 
                                        concat(next_state, goal), done)
            goals = strategy(episode, next_state, n_strategy)
            for g in goals:
                r = self.env.compute_reward(next_state, g, info)
                self.agent.append_to_buffer(concat(state, g), action, r, 
                                            concat(next_state, g), done)
        
        if len(self.agent.replay_buffer) >= self.agent.batch_size:
            for e in range(n_epochs):
                exp = self.agent.replay_buffer.sample()
                self.agent.learn(exp)

        return score

    def test(self, t_max, render=True, num_of_episodes=1):
        concat = self.concat
        score_avg = 0
        for i in range(num_of_episodes):
            state, goal = self.env.reset()
            score = 0
            for j in range(t_max):
                action = self.agent.act(concat(state, goal), test=True)
                if render:
                    self.env.render()
                (state, goal), reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    break
            score_avg += score
            #print(score)
        self.env.close()
        return score_avg/num_of_episodes
