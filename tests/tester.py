import torch
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import LinearSchedule

def train_episode(agent, env, max_t, eps, render=False):
    state = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state, eps)
        if render:
            env.render()
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            return score
    return score

def train_agent(agent, env, max_t, n_episodes, alg_name, eps_start=1.0, 
                eps_end=0.01, eps_horizon=1350, 
                render_freq=None, log=True):
    scores = []  
    scores_window = deque(maxlen=100)
    eps_schd = LinearSchedule(eps_start, eps_end, eps_horizon)
    for i_episode in range(1, n_episodes+1):
        render = (render_freq is not None) and (i_episode % render_freq == 0)
        score = train_episode(agent, env, max_t, eps_schd.value, render)
        scores_window.append(score)
        scores.append(score)
        eps_schd.step()
        
        avg_score = np.mean(scores_window)
        print("\rEpisode %d, AVG. Score %.2f" %(i_episode, avg_score))
        if avg_score >= 199.0:
            print("Solved! Episode %d" %(i_episode))
            fname = "checkpoints/{}.pth".format(alg_name)
            torch.save(agent.online_net.state_dict(), fname)
            break
    env.close()
    return scores

def test_agent(agent, env, max_t, render=True, num_of_episodes=1, log=True):
    for i in range(num_of_episodes):
        state = env.reset()
        score = 0
        for j in range(max_t):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        print(score)
    env.close()
