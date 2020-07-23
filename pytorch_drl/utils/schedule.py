import numpy as np

class Schedule:

    def __init__(self, v_init, v_final, n_steps):
        self.v_init = v_init
        self.v_final = v_final
        self.value = v_init


class LinearSchedule(Schedule):

    def __init__(self, v_init, v_final, n_steps):
        super().__init__(v_init, v_final, n_steps)
        self.delta = (v_final - v_init)/n_steps
        self.op = max if (v_init > v_final) else min

    def step(self):
        self.value = self.op(self.v_final, self.value + self.delta)
        return self.value


class ExpSchedule(Schedule):
    
    def __init__(self, v_init, v_final, n_steps):
        super().__init__(v_init, v_final, n_steps)
        # 1e-3 is ~ 0
        delta = (np.log((1e-3)/abs(v_init - v_final)))/n_steps
        self.delta_exp = np.exp(delta)
        self.exp = 1

    def step(self):
        self.exp = self.exp * self.delta_exp
        self.value = self.v_final + (self.v_init - self.v_final) * self.exp
        return self.value