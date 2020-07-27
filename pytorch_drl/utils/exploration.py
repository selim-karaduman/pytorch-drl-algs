import numpy as np
from pytorch_drl.utils.schedule import LinearSchedule

class OrnsteinUhlenbeck:

    def __init__(self, x_size, mu=0, 
                sigma_init=0.2, sigma_final=0.2, 
                sigma_horizon=1, theta=0.2, dt=1e-2):
        self.mu = mu
        self.x_size = x_size
        self.dt = dt
        self.theta = theta
        self.x = np.zeros(x_size) + mu
        self.sigma = LinearSchedule(sigma_init, sigma_final, sigma_horizon)

    def set(self, x):
        self.x = x

    def step(self):
        dw = np.random.randn(*self.x_size) * np.sqrt(self.dt)
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma.value * dw
        self.x = self.x + dx
        self.sigma.step()
        return self.x

    def reset(self):
        self.x = self.x*0 + self.mu

