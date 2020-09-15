import gym
from gym import spaces
import numpy as np

class BitFlipping(gym.Env):
    
    def __init__(self, num_bits, seed=0):
        super().__init__()
        self.num_bits = num_bits
        self.action_space = spaces.Discrete(num_bits)
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(num_bits,), 
                                            dtype=np.uint8)
        self.reset()

    @property
    def state(self):
        return np.copy(self.string), np.copy(self.target)
    
    def step(self, action):
        self.attemtps += 1
        self.string[action] = 1 - self.string[action]
        if self.attemtps > self.num_bits:
            self.done = True
        reward = self.compute_reward(self.string, self.target, None)
        return self.state, reward, self.done, {}

    def reset(self):
        self.string = np.random.randint(0, 2, self.num_bits)
        self.target = np.random.randint(0, 2, self.num_bits)
        self.attemtps = 0
        self.done = False
        return self.state

    def render(self):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.all(achieved_goal == desired_goal):
            return 0
            self.done = True
        else:
            return -1
