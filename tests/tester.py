import torch
import numpy as np
from collections import deque
from pytorch_drl.utils.schedule import LinearSchedule

def train_episode(agent, env, max_t, eps, render=False):
    state = env.reset()
    score = 0
    for t in range(max_t):
        #print("state:", state)
        action = agent.act(state)
        #print("act:",action)
        if render:
            env.render()
        next_state, reward, done, _ = env.step(action)
        #print("nsstate:",next_state)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            return score
    return score

def train_agent(agent, env, max_t, n_episodes, alg_name, eps_start=1.0, 
                eps_end=0.01, eps_horizon=1350, max_score=195.0,
                render_freq=None, log=True, test_freq=None):
    scores = []  
    scores_window = deque(maxlen=100)
    eps_schd = LinearSchedule(eps_start, eps_end, eps_horizon)
    
    test_scores = []
    test_scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        render = (render_freq is not None) and (i_episode % render_freq == 0)
        score = train_episode(agent, env, max_t, eps_schd.value, render)
        scores_window.append(score)
        scores.append(score)
        eps_schd.step()
        
        avg_score = np.mean(scores_window)

        if (test_freq is not None) and (i_episode % test_freq == 0):
            t_score = test_agent(agent, env, max_t, render=False, num_of_episodes=5, log=False)
            test_scores.append(t_score)
            test_scores_window.append(t_score)
            print("Avg Test score: ", np.mean(test_scores_window))

        print("\rEpisode %d, AVG. Score %.2f" %(i_episode, avg_score))
        if avg_score >= max_score:
            print("Solved! Episode %d" %(i_episode))
            fname = "checkpoints/{}.pth".format(alg_name)
            torch.save(agent.online_net.state_dict(), fname)
            break
    env.close()
    return scores

def test_agent(agent, env, max_t, render=True, num_of_episodes=1, log=True):
    score_avg = 0
    for i in range(num_of_episodes):
        state = env.reset()
        score = 0
        for j in range(max_t):
            action = agent.act(state, test=True)
            if render:
                env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        score_avg += score
        if log:
            print(score)
    env.close()
    return score_avg/num_of_episodes
