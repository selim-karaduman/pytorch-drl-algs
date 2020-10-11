import gym
from gym import spaces
import numpy as np
from gym.envs.classic_control import rendering
from pyglet.window import mouse


RENDER_W = 600
RENDER_H = 400

class Connect4(gym.Env):
    
    def __init__(self, size=(6, 7), k=4, one_player_turn=None, 
                    two_players=False, seed=0):
        super().__init__()
        self.h, self.w = size
        self.k = k
        self.one_player_turn = one_player_turn
        self.two_players = two_players
        self.action_space = spaces.Discrete(self.w)
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(self.h, self.w,), 
                                            dtype=np.uint8)
        self.letter_2_id = {'B': 1, 'R': 2, ' ': 0}
        self.id_2_letter = {1: 'B', 2: 'R', 0: ' '}
        self.viewer = None
        self.reset()

    @property
    def state(self):
        return np.copy(self.board)
    
    def step(self, x):
        if (x >= self.w) or (self.indices[x] < 0) or (self.done):
            return #ignore
        y = self.indices[x]
        self.n_elements += 1
        self.board[y, x] = self.turn+1
        self.indices[x] -= 1
        reward = self.compute_reward(y, x)
        self.turn = (self.turn + 1) % 2
        return self.state, reward, self.done, {}

    def reset(self):
        self.board = np.zeros([self.h, self.w], dtype=np.int32)
        self.indices = np.zeros([self.w], dtype=np.int32) + self.h - 1
        # cache connections
        self.row_connect = np.zeros([self.h, self.w], dtype=np.int32)
        self.col_connect = np.zeros([self.h, self.w], dtype=np.int32)
        self.diagneg_connect = np.zeros([self.h, self.w], dtype=np.int32)
        self.diagpos_connect = np.zeros([self.h, self.w], dtype=np.int32)
        self.turn = 0
        self.n_elements = 0
        self.done = False
        return self.state

    def update_rows(self, y, x):
        self.row_connect[y, x] = 1
        if (x-1 >= 0) and  self.board[y, x-1] == self.board[y, x]:
            l = self.row_connect[y, x-1]
            self.row_connect[y, x - l] += self.row_connect[y, x]
            self.row_connect[y, x] = self.row_connect[y, x - l]
        if (x+1 < self.w) and self.board[y, x+1] == self.board[y, x]:
            l = self.row_connect[y, x+1]
            self.row_connect[y, x + l] += self.row_connect[y, x]
            self.row_connect[y, x] = self.row_connect[y, x + l]
            
            if (x-1 >= 0) and  self.board[y, x-1] == self.board[y, x]:
                l = self.row_connect[y, x-1]
                self.row_connect[y, x - l] = self.row_connect[y, x]
        return self.row_connect[y, x] >= self.k

    def update_cols(self, y, x):
        self.col_connect[y, x] = 1
        if (y+1 < self.h) and self.board[y+1, x] == self.board[y, x]:
            self.col_connect[y, x] += self.col_connect[y+1, x]
        return self.col_connect[y, x] >= self.k

    def update_diagneg(self, y, x):
        self.diagneg_connect[y, x] = 1
        if ((x-1 >= 0) and (y-1 >= 0)
                and (self.board[y-1, x-1] == self.board[y, x])):
            l = self.diagneg_connect[y-1, x-1]
            self.diagneg_connect[y - l, x - l] += self.diagneg_connect[y, x]
            self.diagneg_connect[y, x] = self.diagneg_connect[y - l, x - l]
        if ((x+1 < self.w) and (y+1 < self.h)
                and (self.board[y+1, x+1] == self.board[y, x])):
            l = self.diagneg_connect[y+1, x+1]
            self.diagneg_connect[y + l, x + l] += self.diagneg_connect[y, x]
            self.diagneg_connect[y, x] = self.diagneg_connect[y + l, x + l]
            
            if ((x-1 >= 0) and (y-1 >= 0)
                    and (self.board[y-1, x-1] == self.board[y, x])):
                l = self.diagneg_connect[y-1, x-1]
                self.diagneg_connect[y - l, x - l] = self.diagneg_connect[y, x]
        return self.diagneg_connect[y, x] >= self.k

    def update_diagpos(self, y, x):
        self.diagpos_connect[y, x] = 1
        if ((x+1 < self.w) and (y-1 >= 0)
                and (self.board[y-1, x+1] == self.board[y, x])):
            l = self.diagpos_connect[y-1, x+1]
            self.diagpos_connect[y - l, x + l] += self.diagpos_connect[y, x]
            self.diagpos_connect[y, x] = self.diagpos_connect[y - l, x + l]
        if ((x-1 < self.w) and (y+1 < self.h)
                and (self.board[y+1, x-1] == self.board[y, x])):
            l = self.diagpos_connect[y+1, x-1]
            self.diagpos_connect[y + l, x - l] += self.diagpos_connect[y, x]
            self.diagpos_connect[y, x] = self.diagpos_connect[y + l, x - l]
            
            if ((x+1 < self.w) and (y-1 >= 0)
                and (self.board[y-1, x+1] == self.board[y, x])):
                l = self.diagpos_connect[y-1, x+1]
                self.diagpos_connect[y - l, x + l] = self.diagpos_connect[y, x]
        return self.diagpos_connect[y, x] >= self.k

    def compute_reward(self, y, x):
        # y, x is a free cell
        if (self.update_rows(y, x) or self.update_cols(y, x) 
                or self.update_diagneg(y, x) or self.update_diagpos(y, x)):
            self.done = True
            print("Blue" if self.turn==0 else "Red", "has won")
            return 1
        if self.n_elements == self.w * self.h:
            print("Draw")
            self.done=True
        return 0

    def render(self, mode='human'):
        dw = RENDER_W / self.w
        dh = RENDER_H / self.h
        if self.viewer is None:
            self.viewer = rendering.Viewer(RENDER_W, RENDER_H)
            
            @self.viewer.window.event
            def on_mouse_press(x,y, button, modifier):
                pos_x = int(x//dw)
                if (self.two_players 
                    or (self.one_player_turn is not None 
                        and self.one_player_turn == self.turn)):
                    self.step(pos_x)   

        radius = (min(dw, dh))/2 - 1
        eps_dx = (dw - 2*radius)/2
        eps_dy = (dh - 2*radius)/2
        # draw the frame:
        for i in range(self.w + 1):
            self.viewer.draw_line((dw*i, 0), (dw*i, self.h*dh), 
                color = (0., 0., 0.))

        for i in range(self.h + 1):
            self.viewer.draw_line((0, dh*i), (self.w*dw, dh*i), 
                color = (0., 0., 0.))
        
        for y in range(self.h):
            for x in range(self.w):
                if self.board[y, x] != 0:
                    id = self.board[y, x]
                    if self.id_2_letter[id] == 'B':
                        color = (0.015, 0.015, 0.764)
                    else:
                        color = (0.764, 0.031, 0.015)

                    c = rendering.make_circle(radius=radius, filled=True)
                    c.set_color(*color)
                    dx = dw*x + radius + eps_dx
                    dy = dh * (self.h - 1 - y) + radius + eps_dy
                    trnsf = rendering.Transform(translation=(dx, dy))

                    c.add_attr(trnsf)
                    self.viewer.add_geom(c)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
